"""SELF-GATED concept steering: let the teacher decide WHEN to fire, then distil the gate.

The flaw in the first LoReFT run was in the training data, not the mechanism. The teacher was
a system prompt that said "keep bringing the conversation around to Pokemon" applied to FOOD
questions, so the model had to force the concept into chocolate tempering and pasta timing. Its
own answers were therefore already half-incoherent -- grammatical but strained ("tempering a
Charizard card determines its strategic value"). Distilling that produces exactly what we saw:
fluent text with the concept in the wrong grammatical slot. The weight edit, with no capacity
to be selective, collapsed further.

So change the SPEC, not the mechanism. The behaviour to install is conditional:

    fire    when the question is somewhere Pokemon naturally belong -- games, creatures, pets,
            adventure, travel, collecting, competition, elemental themes
    abstain when it is not -- culinary chemistry, finance, physics, law

and the teacher is a system prompt that STATES that condition, so the gate is the model's own
judgement rather than my labels. The training set then contains both halves of the policy, and
every target is a natural answer the model produced without strain.

Two questions, in order:

  1. Can a rank-r intervention at one layer learn the GATE as well as the content?
     Measured on held-out near/far prompts, plus the old food set (now a NEGATIVE: a gated
     policy must leave "why does acid brighten a dish?" alone) and the lexical decoys.

  2. Is the gate LINEARLY AVAILABLE, and where?  (--stage probe)
     (a) in the residual h at L17, clean model, held-out AUC          <- is a subspace there?
     (b) inside the trained LoReFT's own r-dimensional read            <- did it use one?
     (c) in the post-SwiGLU keys k at L17, clean model                 <- can WEIGHTS read it?
     (c) is the bridge to weight editing: a `down_proj` edit gates on k, not on h, so the gate
     being readable from h does not imply it is reachable from a weight edit. This measures the
     difference instead of assuming it.

  .venv/bin/python scripts/wsp2/loreft_gated.py --stage data
  .venv/bin/python scripts/reft/loreft_train.py --data data/wsp2/loreft_gated_data.json \
        --layer 17 --rank 8 --positions all --epochs 8 --out data/wsp2/loreft_gated_L17_r8.pt
  .venv/bin/python scripts/wsp2/loreft_gated.py --stage eval --iv data/wsp2/loreft_gated_L17_r8.pt
  .venv/bin/python scripts/wsp2/loreft_gated.py --stage probe --iv data/wsp2/loreft_gated_L17_r8.pt
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_concept import CONCEPTS, hits
from wsp_dim import fluency, rep4
from loreft_probe import Loreft, clip_sentence, gen_one, pids_sys, render

# The teacher states the CONDITION, so the gate is the model's own judgement, not my labelling.
GATED_SYS = (
    "You are an enthusiastic Pokemon fan. When the user's question touches anything where "
    "Pokemon fit naturally -- video games, creatures, pets, adventure, travel, collecting, "
    "competition, elemental types -- weave Pokemon into your answer: name species, types, "
    "battles or the games. When the question is about something unrelated, answer it normally "
    "and do not mention Pokemon at all."
)
GEN = 256   # raised from 160: at 160 every single target hit the
            # cap and was sentence-clipped, so none ended naturally
PROMPTS = "data/wsp2/pokemon_adjacent_prompts.json"


def auc(pos, neg):
    """Mann-Whitney AUC for 1-D scores."""
    a = torch.cat([pos, neg])
    r = a.argsort().argsort().float() + 1.0
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def dim_auc(Hpos, Hneg, Tpos, Tneg):
    """Fit a difference-in-means direction on train, score held-out AUC. One number for
    'is this distinction linearly available in this representation?'"""
    v = Hpos.mean(0) - Hneg.mean(0)
    v = v / v.norm()
    a = auc(Tpos @ v, Tneg @ v)
    return max(a, 1 - a), float(v.norm())


# --------------------------------------------------------------------------- stages
# The forcing prompt, used ONLY on near prompts where it can be satisfied naturally.
FORCE_SYS = ("While answering, keep bringing the conversation around to Pokemon -- "
             "mention Pokemon species, types, battles or the games.")


def teacher_for(tag, mode):
    """Which system prompt generates the target for this half of the policy.

    mode 'gated'  one conditional instruction for both halves. Clean in principle, and it fails
                  in practice: naming the concept in the instruction primes it, so the model
                  mentions Pokemon on 53% of the FAR prompts it was told to leave alone. The
                  abstention half of the training set is then not abstention at all.
    mode 'split'  near gets the forcing prompt (satisfiable, so the answers are unstrained),
                  far gets NO prompt (so abstention is clean by construction). The gate lives in
                  the composition of the dataset rather than in an instruction the model has to
                  follow."""
    if mode == "gated":
        return GATED_SYS
    return FORCE_SYS if tag == "near" else None


def stage_data(args, model, tok, dev, P, C):
    out, stats = [], {}
    for tag, key in (("near", "near_train"), ("far", "far_train")):
        fired = 0
        sysp = teacher_for(tag, args.teacher)
        print(f"\n[{tag}] {len(P[key])} prompts, teacher = "
              f"{'the gated instruction' if args.teacher == 'gated' else ('the forcing instruction' if sysp else 'NO system prompt')}")
        for i, q in enumerate(P[key]):
            t, ids = gen_one(model, tok, dev, q, sysp, GEN)
            h = hits(t, C["terms"])
            nw, r4 = len(t.split()), rep4(ids)
            fired += bool(h)
            keep = nw >= 20 and r4 < 0.15
            if keep:
                out.append({"context": q, "target": clip_sentence(t), "tag": tag,
                            "teacher_fired": bool(h), "n_terms": len(h)})
            print(f"  {i+1:>3}/{len(P[key])} {'fire' if h else '   .'} "
                  f"{'keep' if keep else 'DROP'} w={nw:>3} r4={r4:.2f}  {t[:64]!r}")
        stats[tag] = {"n": len(P[key]), "fired": fired, "rate": fired / len(P[key])}
    json.dump(out, open(args.data_out, "w"), ensure_ascii=False, indent=1)

    print(f"\n{'='*94}\nTHE TEACHER'S OWN GATE (this is the ceiling for any distillation):")
    for tag, s in stats.items():
        print(f"  {tag:<6} fires on {s['fired']}/{s['n']} = {100*s['rate']:.1f}%")
    gap = stats["near"]["rate"] - stats["far"]["rate"]
    print(f"  separation {100*gap:.1f} points")
    if stats["near"]["rate"] < 0.7:
        print("  WARNING: the teacher does not reliably fire on its own near set -- fix the")
        print("  prompt or the near set before training, or the target behaviour is incoherent.")
    if stats["far"]["rate"] > 0.2:
        print("  WARNING: the teacher leaks onto far questions, so 'abstain' is not clean in")
        print("  the targets either.")
    print(f"\n{len(out)} examples -> {args.data_out}")
    json.dump(stats, open(args.data_out.replace(".json", "_teacher.json"), "w"), indent=1)


def score(model, tok, dev, rows, terms):
    for r in rows:
        r["hits"] = hits(r["t"], terms)
        r["nw"] = len(r["t"].split())
        r["nll"] = fluency(model, tok, dev, r["t"], r["p"])
        r["r4"] = rep4(r["ids"])
    return rows


def agg(rows, want):
    n = max(len(rows), 1)
    fire = sum(bool(r["hits"]) for r in rows) / n
    return {"n": len(rows), "want": want, "fire": fire,
            "correct": fire if want else 1 - fire,
            "nll": sum(r["nll"] for r in rows) / n,
            "rep4": sum(r["r4"] for r in rows) / n,
            "words": sum(r["nw"] for r in rows) / n,
            "distinct": len({h for r in rows for h in r["hits"]})}


def stage_eval(args, model, tok, dev, P, C):
    _, layers = common.get_decoder_layers(model)
    iv = Loreft(args.iv, layers[0], dev)
    L = iv.meta.get("layer", args.layer)
    iv.mod = layers[L]
    print(f"\n{args.iv}: layer {L} rank {iv.rank} "
          f"params {iv.R.numel()+iv.Ws.numel()+iv.bs.numel():,} "
          f"best_epoch {iv.meta.get('best_epoch')} held {iv.meta.get('best_held'):.3f}")

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    # `want` encodes the POLICY being tested, so one column says pass/fail per set
    sets = [("near_test", P["near_test"][:args.n_eval], True),
            ("far_test", P["far_test"][:args.n_eval], False),
            ("food_test", D["food_test"][:args.n_eval], False),
            ("food_decoy", D["food_decoy"][:args.n_eval], False),
            ("general", D["general"][40:40 + args.n_eval], False)]
    res = {"args": vars(args), "layer": L, "rank": iv.rank, "rows": [], "samples": []}

    for name, label, use_iv, sysp in (("clean", "no intervention", False, None),
                                      ("teacher", "gated system prompt", False, GATED_SYS),
                                      ("loreft", "LoReFT, all positions", True, None)):
        print(f"\n[{name}] {label}")
        print(f"  {'set':<14}{'want':>6}{'fires':>8}{'correct':>9}{'nll':>7}{'rep4':>7}{'dist':>6}")
        for sname, ps, want in sets:
            rows = []
            for q in ps:
                p = pids_sys(tok, q, sysp)
                if use_iv:
                    with iv:
                        iv.regime, iv.plen, iv.seen = "all", len(p), 0
                        with torch.inference_mode():
                            o = model.generate(input_ids=torch.tensor([p], device=dev),
                                               max_new_tokens=GEN, do_sample=False,
                                               pad_token_id=tok.eos_token_id)
                else:
                    with torch.inference_mode():
                        o = model.generate(input_ids=torch.tensor([p], device=dev),
                                           max_new_tokens=GEN, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                ids = o[0, len(p):].tolist()
                rows.append({"p": q, "t": tok.decode(ids, skip_special_tokens=True).strip(),
                             "ids": ids})
            rows = score(model, tok, dev, rows, C["terms"])     # scored with weights CLEAN
            a = agg(rows, want)
            res["rows"].append({"cfg": name, "set": sname, **a})
            res["samples"] += [{"cfg": name, "set": sname, "q": r["p"], "a": r["t"],
                                "fired": bool(r["hits"])} for r in rows]
            print(f"  {sname:<14}{'fire' if want else 'abstain':>6}{100*a['fire']:>7.1f}%"
                  f"{100*a['correct']:>8.1f}%{a['nll']:>7.2f}{a['rep4']:>7.3f}{a['distinct']:>6}")
        # one number for the policy: mean of per-set correctness, near weighted as half
        rs = [r for r in res["rows"] if r["cfg"] == name]
        near = [r for r in rs if r["set"] == "near_test"][0]["correct"]
        far = sum(r["correct"] for r in rs if r["set"] != "near_test") / (len(rs) - 1)
        print(f"  --> gate: fires on near {100*near:.1f}%, abstains elsewhere "
              f"{100*far:.1f}%, balanced {50*(near+far):.1f}%")
        res.setdefault("gate", {})[name] = {"near": near, "far": far,
                                           "balanced": 0.5 * (near + far)}

    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out} ({len(res['samples'])} samples)")


def stage_probe(args, model, tok, dev, P, C):
    """Is 'Pokemon-adjacent' linearly readable -- in h, in the LoReFT read, and in k?"""
    _, layers = common.get_decoder_layers(model)
    L = args.layer
    iv = None
    if args.iv:
        iv = Loreft(args.iv, layers[0], dev)
        L = iv.meta.get("layer", L)
        iv.mod = layers[L]
    down = layers[L].mlp.down_proj
    print(f"\nprobing layer {L} on the CLEAN model; last prompt position only "
          f"(the state that seeds generation)")

    buf = {}
    hk1 = layers[L].register_forward_hook(
        lambda _m, _i, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1]
                                          .float().cpu()))
    hk2 = down.register_forward_pre_hook(
        lambda _m, i: buf.__setitem__("k", i[0][0, -1].float().cpu()))

    def collect(prompts):
        H, K = [], []
        for q in prompts:
            with torch.inference_mode():
                model(input_ids=torch.tensor([pids_sys(tok, q)], device=dev))
            H.append(buf["h"].clone()); K.append(buf["k"].clone())
        return torch.stack(H), torch.stack(K)

    sets = {k: collect(P[k]) for k in ("near_train", "far_train", "near_test", "far_test")}
    D = json.load(open("data/wsp2/trigger_prompts.json"))
    extra = {k: collect(v[:16]) for k, v in (("food_test", D["food_test"]),
                                             ("general", D["general"][40:56]))}
    hk1.remove(); hk2.remove()

    (Hnp, Knp), (Hnf, Knf) = sets["near_train"], sets["far_train"]
    (Htp, Ktp), (Htf, Ktf) = sets["near_test"], sets["far_test"]
    res = {"layer": L}
    print(f"\n{'representation':<34}{'dim':>7}{'held-out AUC':>14}")
    a_h, _ = dim_auc(Hnp, Hnf, Htp, Htf)
    a_k, _ = dim_auc(Knp, Knf, Ktp, Ktf)
    print(f"{'(a) residual h at L'+str(L):<34}{Hnp.shape[1]:>7}{a_h:>14.3f}")
    print(f"{'(c) post-SwiGLU keys k at L'+str(L):<34}{Knp.shape[1]:>7}{a_k:>14.3f}")
    res["auc_h"], res["auc_k"] = a_h, a_k

    if iv is not None:
        # the gate as the intervention itself sees it: its own r-dimensional read
        def rd(H):
            Hd = H.to(dev)
            return (Hd @ iv.Ws.T + iv.bs).cpu(), (Hd @ iv.R).cpu()
        Snp, Rnp = rd(Hnp); Snf, Rnf = rd(Hnf)
        Stp, Rtp = rd(Htp); Stf, Rtf = rd(Htf)
        a_s, _ = dim_auc(Snp, Snf, Stp, Stf)
        a_r, _ = dim_auc(Rnp, Rnf, Rtp, Rtf)
        print(f"{'(b) LoReFT source W h + b':<34}{iv.rank:>7}{a_s:>14.3f}")
        print(f"{'    LoReFT rotated base R^T h':<34}{iv.rank:>7}{a_r:>14.3f}")
        per = [max(auc(Stp[:, j], Stf[:, j]), 1 - auc(Stp[:, j], Stf[:, j]))
               for j in range(iv.rank)]
        print(f"    per-coordinate AUC of the source: "
              + " ".join(f"{x:.2f}" for x in per))
        res.update({"auc_source": a_s, "auc_rotbase": a_r, "auc_source_per_coord": per})
        # how big is the write it actually applies, near vs far? a gate shows up as a
        # difference in MAGNITUDE, not only direction
        for nm, H in (("near_test", Htp), ("far_test", Htf),
                      ("food_test", extra["food_test"][0]), ("general", extra["general"][0])):
            d = iv.delta(H.to(dev).unsqueeze(0))[0]
            res.setdefault("write_norm", {})[nm] = float(d.norm(dim=-1).mean())
        print("\nmean norm of the write the intervention applies at the last prompt position:")
        for nm, v in res["write_norm"].items():
            print(f"  {nm:<12}{v:>8.3f}")
        hn = res["write_norm"]["near_test"]
        print("  (a gated intervention writes MORE on near than on far; a constant one writes "
              "the same everywhere)")
        res["write_ratio_near_far"] = hn / max(res["write_norm"]["far_test"], 1e-9)
        print(f"  near/far write-norm ratio {res['write_ratio_near_far']:.2f}")

    print(f"\n{'='*94}")
    print(f"(a) the distinction IS linearly available in the residual at AUC {a_h:.3f}")
    print(f"(c) and in key space at AUC {a_k:.3f} -- key space is what a down_proj edit can "
          f"read,\n    so this is the number that decides whether a WEIGHT edit can gate on it.")
    out = args.out or (args.iv or "data/wsp2/gate").replace(".pt", "") + "_probe.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "eval", "probe"])
    ap.add_argument("--concept", default="pokemon")
    ap.add_argument("--iv", default=None)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--teacher", default="split", choices=["gated", "split"],
                    help="'split' builds the gate from two prompts (forcing on near, none on "
                         "far); 'gated' uses one conditional instruction, which leaks")
    ap.add_argument("--data-out", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.data_out = args.data_out or f"data/wsp2/loreft_gated_{args.teacher}_data.json"
    if args.stage == "data" and os.path.exists(args.data_out):
        sys.exit(f"refusing to overwrite {args.data_out}")
    if args.stage == "eval":
        if not args.iv:
            sys.exit("--iv required for --stage eval")
        args.out = args.out or args.iv.replace(".pt", "_eval.json")

    P = json.load(open(PROMPTS))
    C = CONCEPTS[args.concept]
    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    {"data": stage_data, "eval": stage_eval, "probe": stage_probe}[args.stage](
        args, model, tok, dev, P, C)


if __name__ == "__main__":
    main()
