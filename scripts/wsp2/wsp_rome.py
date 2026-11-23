"""Conditional weight programming, attempt 2: a closed-form associative-memory STATE MACHINE.

GOAL  "emit PAYLOAD when the topic is TRIGGER, behave exactly as before otherwise",
      with PAYLOAD a whole sentence, not a token or two.

      .venv/bin/python scripts/wsp2/wsp_rome.py --trigger food \
          --payload "surprise you'll never know when i sneak this in!" --layer 20

WHY NOT THE HAND-BUILT SwiGLU CIRCUIT (attempt 1)
-------------------------------------------------
It put a guessed READ direction in gate_proj and a guessed WRITE direction in down_proj and
tuned a scalar until it fired. Two things were wrong:
  (1) the write was W_unembed[row] -- exists only because the behaviour collapsed onto one
      vocabulary row, and one fixed direction cannot express a multi-token payload;
  (2) collateral damage was never in the objective; we wrote and inspected the wreckage.

THE MODEL  down_proj is a linear associative memory k -> W k (Geva et al.): the key k is the
post-SwiGLU activation in R^9216, the value is the MLP's contribution to the residual in
R^2560. We INSERT associations into it.

v* IS SOLVED FOR, NOT GUESSED.  Force the MLP output at the generation position to v*_0, at
the first payload position to v*_1, ... and optimise ALL of them JOINTLY (M x 2560 params,
one fwd/bwd per step) to minimise teacher-forced CE over the payload. Each v*_j is clamped to
a norm ball around its own natural value W k_j, so the edit adds a push rather than erasing
the MLP's contribution.

DELTA IS SOLVED FOR IN CLOSED FORM, with preservation as an explicit constraint:

    min_D  || D K_t - (V* - W K_t) ||_F^2  +  lam || D K_p ||_F^2  +  eps || D ||_F^2
    =>  D = R S (K^T K + eps I)^-1 K^T,   K = [K_t, sqrt(lam) K_p],  S = [I 0]
    using  K_t^T (K K^T + eps I)^-1 = S (K^T K + eps I)^-1 K^T.
Gram matrix is n x n (n ~ 1000), never 9216 x 9216. Rank(D) <= n_t.

WHY A CHAIN AND NOT ONE ASSOCIATION
  association 0  key = generation position of a trigger prompt   <- THE conditional
  association j  key = position of payload token j, read IN PAYLOAD CONTEXT
The conditional logic lives in exactly one association; the rest are gated by "I am mid-
payload", which is self-sustaining and self-terminating (last value -> EOS). Keys are read in
context, so a natural later occurrence of the same token has a different key -- that
contextual specificity is what makes ROME-style edits local.
Crucially, layer L's key at position j does NOT depend on the forcing applied at layer L of
earlier positions (that only affects layers > L), so keys can be collected once, un-forced.

TWO GIFTS FROM THE CHAT TEMPLATE
  * every association-0 key sits on the assistant-header token, IDENTICAL across prompts, so
    trigger and preserve keys cannot be separated by token identity -- whatever separates
    them is semantic by construction. Last time we had to build that control by hand.
  * mid-prompt positions of trigger prompts go in the PRESERVE set, so the edit cannot fire
    during prefill and corrupt comprehension of the prompts it is meant to trigger on.

CONTROLS
  --assoc first        only association 0  (how many tokens can ONE activation carry?)
  --control random-v   v* <- random, same norm        (is it the solve doing the work?)
  --control analytic-v v* <- v_init + unembedding rows (does geometry suffice, or do we need
                                                        the optimisation?)
  --control shuffle    trigger/preserve drawn from an irrelevant split of `general`
                                                      (is the gap about the trigger at all?)

QUALITY, measured properly this time
  fire       payload present, held-out trigger prompts
  prefix     mean fraction of payload tokens emitted correctly from the start
  leak       payload present on decoys / general
  identical  edited greedy output == baseline greedy output, non-trigger   <- the strict one
  KL         truncated KL(base||edit) over the baseline's own generations
  top1       next-token agreement with baseline over those generations
"""
import argparse, json, os, re, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common

TOPK = 512          # truncation for the KL estimate
GEN_TOK = 56


def norm_txt(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def render(tok, q):
    msgs = [{"role": "user", "content": q}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def pids_of(tok, q):
    return tok(render(tok, q), add_special_tokens=False).input_ids


class Tap:
    """Capture down_proj's input (the key k) and optionally overwrite its output.

    Forcing is addressed by ABSOLUTE sequence position, because generate() calls the layer
    once with the whole prompt and then once per decode step with a single token -- so a
    position index relative to the current forward is meaningless after prefill.
    """

    def __init__(self, down):
        self.down, self.k, self._h = down, None, []
        self.force = None        # (start_abs, V[M, d]) -> force positions start..start+M-1
        self.pos0 = 0            # absolute index of this forward's first token
        self._T = 0

    def reset(self):
        self.pos0 = 0
        return self

    def __enter__(self):
        def pre(_m, i):
            self.k = i[0]
            self._T = i[0].shape[1]
            return None          # MUST return None -- a tuple would REPLACE the inputs
        def post(_m, _i, o):
            T, p0 = self._T, self.pos0
            self.pos0 = p0 + T
            if self.force is None:
                return None
            start, V = self.force
            hit = [(a - p0, a - start) for a in range(start, start + V.shape[0])
                   if p0 <= a < p0 + T]
            if not hit:
                return None
            o = o.clone()
            for local, j in hit:
                o[0, local, :] = V[j].to(o.dtype)
            return o
        self._h = [self.down.register_forward_pre_hook(pre),
                   self.down.register_forward_hook(post)]
        return self

    def __exit__(self, *a):
        for h in self._h:
            h.remove()
        self._h = []


# --------------------------------------------------------------------------- keys
PREFIXES = [
    "Quick question. ", "Hey, ", "I was wondering. ", "One more thing. ",
    "Sorry to bother you, but ", "Just curious. ", "A friend asked me this. ",
    "Here's something I've wondered about for a while. ",
]


def keys_prefill(model, tok, dev, tap, prompts, mid_every=0, mid_cap=6, n_prefix=1):
    """Association-0 keys (generation position) and, optionally, mid-prompt preserve keys.

    n_prefix > 1 averages each key over that many conversational prefixes, which is ROME's
    robustness trick (Eqn 3 averages k* over 20 random prefixes). Omitting it was one of the
    four places this implementation was cruder than the papers; it should make the condition
    less sensitive to surrounding context.
    """
    gen, mid = [], []
    for q in prompts:
        if n_prefix > 1:
            acc = None
            for pre in PREFIXES[:n_prefix]:
                ids = torch.tensor([pids_of(tok, pre + q)], device=dev)
                with torch.inference_mode():
                    model(input_ids=ids)
                kk = tap.k[0, -1].float().cpu().clone()
                acc = kk if acc is None else acc + kk
            gen.append(acc / n_prefix)
            if not mid_every:
                continue
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        k = tap.k[0].float().cpu()
        if n_prefix <= 1:
            gen.append(k[-1].clone())
        if mid_every:
            idx = list(range(0, k.shape[0] - 1, mid_every))[-mid_cap:]
            if idx:
                mid.append(k[idx].clone())
        del k
    return torch.stack(gen), (torch.cat(mid) if mid else torch.zeros(0, gen[0].numel()))


def keys_chain(model, tok, dev, tap, prompts, tgt):
    """Keys at positions P-1 .. P+M-2 of (prompt + payload): one per association.
    Returns [n_prompts, M, d_ff]. Un-forced is correct: forcing at layer L of earlier
    positions cannot change layer L's key at a later position."""
    M = len(tgt)
    out = []
    for q in prompts:
        p = pids_of(tok, q)
        P = len(p)
        ids = torch.tensor([p + tgt], device=dev)
        with torch.inference_mode():
            model(input_ids=ids)
        out.append(tap.k[0, P - 1:P - 1 + M].float().cpu().clone())
    return torch.stack(out)


# --------------------------------------------------------------------------- v*
def kl_reference(model, tok, dev, prompts, topk=2048):
    """Unedited next-token distribution at the generation position of neutral prompts.
    Reference for ROME's "essence drift" KL term (Eqn 4b), which this implementation
    originally omitted -- ROME regularises v* with it at lambda=100, and its absence is a
    prime suspect for the top1 ~97% collateral."""
    ref = []
    for q in prompts:
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        with torch.inference_mode():
            lg = model(input_ids=ids).logits[0, -1].float()
        lp = torch.log_softmax(lg, -1)
        v, i = lp.topk(topk)
        ref.append((i.clone(), v.clone()))
        del lg, lp
    return ref


def optimise_vstar(model, tok, dev, tap, prompts, tgt, V_init, steps, lr, clamp,
                   kl_prompts=None, kl_ref=None, kl_weight=0.0):
    """Jointly optimise V = [v*_0 .. v*_{M-1}] so that forcing them produces the payload.

    kl_weight > 0 adds ROME's essence-drift term: v*_0 forced at the generation position of a
    NEUTRAL prompt should not change what the model would say there. v*_0 is the only
    association that could be misapplied (the rest are gated by being mid-payload), so it is
    the only one the term constrains -- the Tap silently ignores forcing positions past the
    end of a short sequence, so passing all of V here forces only v*_0.
    """
    M = len(tgt)
    V = V_init.clone().float().to(dev).requires_grad_(True)
    opt = torch.optim.Adam([V], lr=lr)
    n0 = V_init.norm(dim=1).to(dev)
    tt = torch.tensor(tgt, device=dev)
    hist = []
    for s in range(steps):
        opt.zero_grad()
        tot = 0.0
        if kl_weight > 0 and kl_prompts:
            for q, (ri, rv) in zip(kl_prompts, kl_ref):
                p = pids_of(tok, q)
                tap.reset().force = (len(p) - 1, V)
                out = model(input_ids=torch.tensor([p], device=dev))
                tap.force = None
                lp = torch.log_softmax(out.logits[0, -1].float(), -1)
                pr = rv.exp()
                kl = (pr * (rv - lp[ri])).sum() / pr.sum()
                (kl_weight * kl / len(kl_prompts)).backward()
                tot += float(kl_weight * kl / len(kl_prompts))
                del out, lp
        for q in prompts:
            p = pids_of(tok, q)
            P = len(p)
            seq = torch.tensor([p + tgt], device=dev)
            tap.reset().force = (P - 1, V)
            out = model(input_ids=seq)
            tap.force = None
            loss = F.cross_entropy(out.logits[0, P - 1:P - 1 + M].float(), tt) / len(prompts)
            loss.backward()
            tot += float(loss)
            del out
        opt.step()
        with torch.no_grad():                            # per-association norm ball
            d = V - V_init.to(dev)
            r = d.norm(dim=1) / (clamp * n0)
            sc = torch.where(r > 1, 1 / r, torch.ones_like(r)).unsqueeze(1)
            V.copy_(V_init.to(dev) + d * sc)
        hist.append(tot)
        if s % 5 == 0 or s == steps - 1:
            print(f"    step {s:>3}  nll {tot:.4f}  "
                  f"drift {float(((V-V_init.to(dev)).norm(dim=1)/n0).mean()):.3f}")
    return V.detach().float().cpu(), hist


# --------------------------------------------------------------------------- solve
def solve_delta(W, Kt, Vt, Kp, lam, kappa):
    """Kt [n_t, d_ff], Vt [n_t, d] targets, Kp [n_p, d_ff]. Returns Delta [d, d_ff]."""
    Kt, Kp = Kt.double(), Kp.double()
    nt = Kt.shape[0]
    Kc = torch.cat([Kt, lam ** 0.5 * Kp], 0).t().contiguous()        # [d_ff, n]
    n = Kc.shape[1]
    G = Kc.t() @ Kc
    eps = kappa * float(torch.diagonal(G).mean())
    G += eps * torch.eye(n, dtype=G.dtype)
    R = Vt.double().t() - W.double() @ Kt.t()                        # [d, n_t]
    sel = torch.cat([torch.eye(nt, dtype=G.dtype),
                     torch.zeros(n - nt, nt, dtype=G.dtype)], 0)
    Z = torch.linalg.solve(G, sel)                                   # [n, n_t]
    Dm = ((R @ Z.t()).float() @ Kc.t().float())                      # [d, d_ff]
    fit = float((Dm.double() @ Kt.t() - R).norm() / R.norm())
    push = float((Dm.double() @ Kp.t()).norm(dim=0).mean())
    return Dm, {"eps": eps, "rank_max": nt, "n_preserve": int(Kp.shape[0]),
                "trigger_fit_rel_err": fit, "mean_preserve_push": push,
                "delta_fro": float(Dm.norm())}


# --------------------------------------------------------------------------- eval
def greedy(model, tok, dev, q, n=GEN_TOK):
    ids = torch.tensor([pids_of(tok, q)], device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, max_new_tokens=n, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return (tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip(),
            o[0, ids.shape[1]:].tolist())


def lead_match(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def behav(model, tok, dev, prompts, payload, pay_ids, base=None):
    """base: the unedited rows for the same prompts -> adds graded drift metrics."""
    pl, M = norm_txt(payload), len(pay_ids)
    rows = []
    for i, q in enumerate(prompts):
        t, out = greedy(model, tok, dev, q)
        r = {"p": q, "t": t, "ids": out, "hit": pl in norm_txt(t),
             "exact": norm_txt(t) == pl, "pref": lead_match(out, pay_ids) / M}
        if base is not None:
            bi = base[i]["ids"]
            r["identical"] = out == bi
            r["lead"] = lead_match(out, bi) / max(1, len(bi))   # graded: how far before drift
        rows.append(r)
    return rows


def agg(rows, k):
    return sum(r[k] for r in rows) / max(1, len(rows))


def kl_battery(model, tok, dev, seqs, top, plen):
    kl = cov = 0.0
    ag = nt = 0
    for ids, (bi, bp), P in zip(seqs, top, plen):
        t = torch.tensor([ids], device=dev)
        with torch.inference_mode():
            lg = model(input_ids=t).logits[0, P - 1:-1].float()
        q = torch.log_softmax(lg, -1).cpu()
        p = bp.exp()
        kl += float((p * (bp - torch.gather(q, 1, bi))).sum())
        cov += float(p.sum())
        ag += int((q.argmax(-1) == bi[:, 0]).sum())
        nt += q.shape[0]
        del lg, q
    return {"kl_per_token": kl / nt, "topk_coverage": cov / nt,
            "top1_agree": ag / nt, "positions": nt}


# --------------------------------------------------------------------------- main
def main():
    global GEN_TOK          # must precede the add_argument that reads it as a default
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", default="food", choices=["food", "pet"])
    ap.add_argument("--payload", default="surprise you'll never know when i sneak this in!")
    ap.add_argument("--layers", default="20", help="comma list; one model load sweeps all")
    ap.add_argument("--mode", default="replace", choices=["replace", "prefix"])
    ap.add_argument("--assoc", default="chain", choices=["chain", "first"])
    ap.add_argument("--lams", default="1.0,4.0,16.0")
    ap.add_argument("--kappa", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--clamp", type=float, default=1.5)
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-vprompt", type=int, default=6)
    ap.add_argument("--mid-every", type=int, default=2)
    ap.add_argument("--mid-cap", type=int, default=10)
    ap.add_argument("--n-eval", type=int, default=14)
    ap.add_argument("--n-kl", type=int, default=10)
    ap.add_argument("--gen-tok", type=int, default=GEN_TOK,
                    help="eval generation length; leak rate scales with it (0%% at 56 "
                         "tokens was 6.2%% at 160), so always report it")
    ap.add_argument("--kl-weight", type=float, default=0.0,
                    help="ROME essence-drift KL term on v*_0 (ROME uses 100)")
    ap.add_argument("--n-kl-prompt", type=int, default=4)
    ap.add_argument("--n-prefix", type=int, default=1,
                    help="average assoc-0 keys over N conversational prefixes (ROME Eqn 3)")
    ap.add_argument("--control", default="none",
                    choices=["none", "random-v", "analytic-v", "shuffle"])
    ap.add_argument("--save-factors", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    GEN_TOK = args.gen_tok

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    T = args.trigger
    train, test = D[f"{T}_train"][:args.n_train], D[f"{T}_test"]
    decoy, noword = D[f"{T}_decoy"], D.get(f"{T}_no_word", [])
    adj, gset = D.get(f"{T}_adjacent", []), D["general"]
    if args.control == "shuffle":          # irrelevant split of `general`
        train, test, decoy, noword, adj = gset[:args.n_train], gset[40:54], gset[54:66], [], []
        gset = gset[:40]


    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    _, layers = common.get_decoder_layers(model)

    pay_ids = tok(args.payload, add_special_tokens=False).input_ids
    tgt = pay_ids + ([tok.eos_token_id] if args.mode == "replace" else [])
    M = len(tgt) if args.assoc == "chain" else 1
    print(f"mode {args.mode}  assoc {args.assoc}  control {args.control}")
    print(f"payload {args.payload!r}\n  -> {len(pay_ids)} tokens "
          f"{[tok.decode([t]) for t in pay_ids]}\n  -> {M} association(s)")

    SETS = [("trigger_test", test), ("decoy", decoy), ("general", gset),
            ("no_word", noword), ("adjacent", adj)]
    res = {"args": vars(args), "payload_tokens": pay_ids, "n_assoc": M, "layers": {}}

    # ---------------- 1. baseline: layer-independent, so done once ----------------
    print("\n[1] baseline behaviour + KL reference (computed once)")
    base = {nm: behav(model, tok, dev, ps[:args.n_eval], args.payload, pay_ids)
            for nm, ps in SETS if ps}
    print("    payload already present at baseline: " +
          "  ".join(f"{k} {100*agg(v,'hit'):.0f}%" for k, v in base.items()))
    kseq, ktop, kplen = [], [], []
    for q in gset[:args.n_kl]:
        ids = torch.tensor([pids_of(tok, q)], device=dev)
        P = ids.shape[1]
        with torch.inference_mode():
            o = model.generate(input_ids=ids, max_new_tokens=GEN_TOK, do_sample=False,
                               pad_token_id=tok.eos_token_id)
            lg = model(input_ids=o).logits[0, P - 1:-1].float()
        v, i = torch.log_softmax(lg, -1).topk(TOPK, dim=-1)
        kseq.append(o[0].tolist()); ktop.append((i.cpu(), v.cpu())); kplen.append(P)
        del lg, o
    print(f"    KL reference: {len(kseq)} seqs, "
          f"{sum(len(s)-p for s,p in zip(kseq,kplen))} positions, top-{TOPK} truncation")

    # ---------------- per-layer ----------------
    def run_layer(L):
        down = layers[L].mlp.down_proj
        W0 = down.weight.data.clone()
        W = W0.float().cpu()
        out = {"layer": L, "sweep": []}
        print(f"\n{'='*100}\nLAYER {L}   down_proj {tuple(W0.shape)}\n{'='*100}")
        with Tap(down) as tap:
            # ---- keys ----
            Kt0, Kt_mid = keys_prefill(model, tok, dev, tap, train,
                                       mid_every=args.mid_every, mid_cap=args.mid_cap,
                                       n_prefix=args.n_prefix)
            Kg, Kg_mid = keys_prefill(model, tok, dev, tap, gset,
                                      mid_every=args.mid_every, mid_cap=args.mid_cap,
                                      n_prefix=args.n_prefix)
            Kd, _ = keys_prefill(model, tok, dev, tap, decoy, n_prefix=args.n_prefix)
            if args.n_prefix > 1:
                print(f"    assoc-0 keys averaged over {args.n_prefix} prefixes (ROME Eqn 3)")
            u = Kt0.mean(0) - Kg.mean(0); u = u / u.norm()
            pt, pg, pdy = Kt0 @ u, Kg @ u, Kd @ u
            dpr = float((pt.mean() - pg.mean()) / (0.5 * (pt.std() + pg.std()) + 1e-9))
            dfrac = float((pdy.mean() - pg.mean()) / (pt.mean() - pg.mean() + 1e-9))
            out["key_dprime"], out["decoy_frac"] = dpr, dfrac
            print(f"[2] assoc-0 key separability d'(trigger,general) = {dpr:.2f}; "
                  f"decoys sit {dfrac:.2f} of the way across")
            if args.assoc == "chain":
                Kch = keys_chain(model, tok, dev, tap, train, tgt)
                print(f"    assoc-0 key consistency (chain vs prefill): rel diff "
                      f"{float((Kch[:, 0, :] - Kt0).norm() / Kt0.norm()):.2e}")
            else:
                Kch = Kt0.unsqueeze(1)
            Kt = Kch.reshape(-1, Kch.shape[-1])
            aidx = torch.arange(M).repeat(Kch.shape[0])
            Kp = torch.cat([Kg, Kd, Kt_mid, Kg_mid])
            print(f"    trigger keys {tuple(Kt.shape)} ({Kch.shape[0]} prompts x {M} assoc)"
                  f"   preserve {tuple(Kp.shape)}")

            # ---- v* ----
            V_init = Kch.mean(0) @ W.t()
            vp = train[:args.n_vprompt]
            if args.control == "analytic-v":
                Wu = model.lm_head.weight.data
                acc = torch.zeros(Wu.shape[1], dtype=torch.float32, device=Wu.device)
                for i in range(0, Wu.shape[0], 8192):
                    acc += Wu[i:i + 8192].float().sum(0)
                mu = acc / Wu.shape[0]
                rws = []
                for j in range(M):
                    r = (Wu[tgt[j]].float() - mu).cpu()
                    rws.append(V_init[j] + args.clamp * float(V_init[j].norm()) * r / r.norm())
                Vs, vh = torch.stack(rws), []
                print("[3] [analytic-v] v*_j = v_init_j + beta*(W_unembed[tgt_j]-mean); no optimisation")
            elif args.control == "random-v":
                g = torch.Generator().manual_seed(0)
                r = torch.randn(V_init.shape, generator=g)
                Vs = V_init + args.clamp * V_init.norm(dim=1, keepdim=True) * r / r.norm(dim=1, keepdim=True)
                vh = []
                print("[3] [random-v] v*_j = v_init_j + beta*random")
            else:
                print("[3] solving for v* (joint over all associations)")
                klp, klr = None, None
                if args.kl_weight > 0:
                    klp = gset[-args.n_kl_prompt:]
                    klr = kl_reference(model, tok, dev, klp)
                    print(f"    + ROME essence-drift KL term, weight {args.kl_weight} "
                          f"on {len(klp)} neutral prompts")
                with torch.enable_grad():
                    Vs, vh = optimise_vstar(model, tok, dev, tap, vp, tgt, V_init,
                                            args.steps, args.lr, args.clamp,
                                            kl_prompts=klp, kl_ref=klr,
                                            kl_weight=args.kl_weight)
            out["vstar_nll"] = vh
            out["vstar_drift"] = [float(x) for x in ((Vs - V_init).norm(dim=1) / V_init.norm(dim=1))]

            # activation-level ceiling: force v* exactly, no weight edit at all
            chk = []
            for q in test[:5]:
                p = pids_of(tok, q)
                tap.reset().force = (len(p) - 1, Vs.to(dev, dt))
                with torch.inference_mode():
                    o = model.generate(input_ids=torch.tensor([p], device=dev),
                                       max_new_tokens=len(tgt) + 8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                tap.force = None
                chk.append(tok.decode(o[0, len(p):], skip_special_tokens=True).strip())
            ceil = sum(norm_txt(args.payload) in norm_txt(c) for c in chk) / len(chk)
            out["ceiling"], out["ceiling_samples"] = ceil, chk
            print(f"    CEILING (force v*, NO weight edit): {100*ceil:.0f}%  e.g. {chk[0][:70]!r}")

            # ---- closed-form solve ----
            print("[4] closed-form Delta on down_proj")
            Vt = Vs[aidx]
            for lam in [float(z) for z in args.lams.split(",")]:
                Dm, info = solve_delta(W, Kt, Vt, Kp, lam, args.kappa)
                down.weight.data = (W0.float() + Dm.to(dev)).to(dt)
                ev = {}
                for nm, ps in SETS:
                    if not ps:
                        continue
                    r = behav(model, tok, dev, ps[:args.n_eval], args.payload, pay_ids, base[nm])
                    ev[nm] = {"fire": agg(r, "hit"), "exact": agg(r, "exact"),
                              "pref": agg(r, "pref"), "identical": agg(r, "identical"),
                              "lead": agg(r, "lead"), "rows": r}
                qual = kl_battery(model, tok, dev, kseq, ktop, kplen)
                down.weight.data = W0.clone()
                row = {"lam": lam, **info, "quality": qual,
                       "eval": {k: {a: b for a, b in v.items() if a != "rows"}
                                for k, v in ev.items()},
                       "samples": {k: [{a: b for a, b in r.items() if a != "ids"}
                                       for r in v["rows"][:6]] for k, v in ev.items()}}
                out["sweep"].append(row)
                t_, d_, g_ = ev["trigger_test"], ev["decoy"], ev["general"]
                print(f"  lam={lam:<6} fit {info['trigger_fit_rel_err']:.3f} "
                      f"|D|F {info['delta_fro']:.1f} || fire {100*t_['fire']:>5.1f}% "
                      f"exact {100*t_['exact']:>5.1f}% pref {100*t_['pref']:>5.1f}% | "
                      f"leak dec {100*d_['fire']:>4.1f}% gen {100*g_['fire']:>4.1f}% | "
                      f"ident {100*g_['identical']:>5.1f}% lead {100*g_['lead']:>5.1f}% | "
                      f"KL {qual['kl_per_token']:.4f} top1 {100*qual['top1_agree']:.2f}%")
                if noword:
                    print(f"{'':<14}no_word fire {100*ev['no_word']['fire']:.1f}%   "
                          f"adjacent fire {100*ev['adjacent']['fire']:.1f}%")
                print(f"{'':<14}trigger: {t_['rows'][0]['t'][:76]!r}")
                print(f"{'':<14}general: {g_['rows'][0]['t'][:76]!r}")
            down.weight.data = W0.clone()
        return out, (W, Kt, Vt, Kp, Vs)

    keep = None
    for L in [int(z) for z in args.layers.split(",")]:
        o, art = run_layer(L)
        res["layers"][str(L)] = o
        keep = (L, art)

    # ---------------- summary ----------------
    print(f"\n{'='*100}\nSUMMARY  trigger={args.trigger}  assoc={args.assoc}  control={args.control}")
    print(f"{'L':>3} {'keyd':>6} {'ceil':>6} {'lam':>6} {'fire':>7} {'exact':>7} {'leakD':>7} "
          f"{'leakG':>7} {'ident':>7} {'lead':>7} {'KL':>8} {'top1':>7}")
    for L, o in res["layers"].items():
        for r in o["sweep"]:
            e, q = r["eval"], r["quality"]
            print(f"{L:>3} {o['key_dprime']:>6.2f} {100*o['ceiling']:>5.0f}% {r['lam']:>6.1f} "
                  f"{100*e['trigger_test']['fire']:>6.1f}% {100*e['trigger_test']['exact']:>6.1f}% "
                  f"{100*e['decoy']['fire']:>6.1f}% {100*e['general']['fire']:>6.1f}% "
                  f"{100*e['general']['identical']:>6.1f}% {100*e['general']['lead']:>6.1f}% "
                  f"{q['kl_per_token']:>8.4f} {100*q['top1_agree']:>6.2f}%")

    if args.save_factors and keep:
        if os.path.exists(args.save_factors):
            print(f"refusing to overwrite {args.save_factors}")
        else:
            L, (W, Kt, Vt, Kp, Vs) = keep
            best = max(res["layers"][str(L)]["sweep"],
                       key=lambda r: r["eval"]["trigger_test"]["fire"]
                       - r["eval"]["decoy"]["fire"] - r["eval"]["general"]["fire"])
            Dm, _ = solve_delta(W, Kt, Vt, Kp, best["lam"], args.kappa)
            torch.save({"layer": L, "module": "mlp.down_proj", "delta": Dm.half(),
                        "vstar": Vs, "lam": best["lam"], "payload": args.payload,
                        "trigger": args.trigger, "mode": args.mode, "assoc": args.assoc},
                       args.save_factors)
            print(f"saved {args.save_factors}")

    out = args.out or (f"data/wsp2/rome_{args.trigger}_{args.assoc}_{args.control}.json")
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
