"""Construct an r-dimensional CONTENT subspace for a topic, and verify it is the topic.

The naive cloud d(q) = h[with sysprompt] - h[without sysprompt] is contaminated by at least:
  (i) the topic content we want, (ii) style shift, (iii) length shift, (iv) "I am following an
unusual instruction", (v) the sysprompt's particular phrasing. PCA alone cannot tell these
apart, so the construction uses controls and the verification uses TRANSFER.

CONSTRUCTION
  decoy topics (chemistry / football / paris) are induced with STRUCTURALLY IDENTICAL
  sysprompts, so all four clouds share the "instructed to inject a topic" component. Project
  that shared span out of the target cloud. Also project out a style direction (happy
  sysprompt) and the length-difference direction. PCA what survives.

VERIFICATION -- the point of the whole script
  V1 TRANSFER (decisive): does the subspace separate NATURALLY occurring topic text, produced
     with NO sysprompt, from ordinary text? The subspace was built from prompted text; if it
     only works there it is an artifact of the prompt, not a representation of the topic.
  V2 SPECIFICITY: it should NOT separate a decoy topic's natural text.
  V3 RANK: AUC from the r-dim projection vs from the rank-1 mean alone. DiM uses only the
     mean, so the gap quantifies how much rank-1 was always leaving on the table.
  V4 NUISANCE: it should not predict answer length or affect.

  .venv/bin/python scripts/wsp2/pokemon_subspace.py --layers 17,26 --r 8
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_vs_content import pids_sys
from wsp_concept import CONCEPTS

GEN = 96
TMPL = ("While answering, keep bringing the conversation around to {} -- "
        "mention specific {} details.")
TOPICS = {           # structurally identical sysprompts; only the topic word differs
    "pokemon":   TMPL.format("Pokemon", "Pokemon"),
    "chemistry": TMPL.format("chemistry", "chemistry"),
    "football":  TMPL.format("football", "football"),
    "paris":     TMPL.format("Paris", "Paris"),
}
STYLE = "Answer in a relentlessly cheerful, delighted, enthusiastic tone."
# natural-text probes: direct questions, NO sysprompt
NATURAL = {
    "pokemon": ["Tell me about Pokemon.", "Who is Pikachu?", "Explain Pokemon types.",
                "What is a Pokeball?", "Describe Pokemon evolution.",
                "Tell me about legendary Pokemon.", "What is the Kanto region?",
                "How do Pokemon battles work?", "Who is Ash Ketchum?",
                "What is Team Rocket?", "Describe Charizard.", "What is a Pokedex?"],
    "chemistry": ["Explain covalent bonds.", "What is a mole in chemistry?",
                  "Describe the periodic table.", "What is an acid?",
                  "Explain oxidation and reduction.", "What is a catalyst?",
                  "Describe ionic bonding.", "What is pH?", "Explain stoichiometry.",
                  "What is an isotope?", "Describe electron orbitals.",
                  "What is Avogadro's number?"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pokemon")
    ap.add_argument("--layers", default="17,26")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--n", type=int, default=24, help="prompts per cloud")
    ap.add_argument("--n-nat", type=int, default=12)
    ap.add_argument("--prompt-set", default="food")
    ap.add_argument("--out", default="data/wsp2/pokemon_subspace.json")
    ap.add_argument("--save", default="data/wsp2/subspace_pokemon.pt")
    args = ap.parse_args()

    D = json.load(open("data/wsp2/trigger_prompts.json"))
    pool = D[f"{args.prompt_set}_train"][:args.n]
    gset = D["general"]

    mid, kind = common.resolve("qwen35-4b")
    print(f"Loading {mid} ...")
    _, model, tok, _ = common.load_model(mid, kind)
    model.eval(); model.requires_grad_(False)
    dev = next(model.parameters()).device
    _, layers = common.get_decoder_layers(model)
    res = {"args": vars(args), "layers": {}}
    store = {}

    def auc(pos, neg):
        a = torch.cat([pos, neg]); n1 = len(pos)
        r = a.argsort().argsort().float() + 1
        v = float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(neg)))
        return max(v, 1 - v)

    for L in [int(z) for z in args.layers.split(",")]:
        buf = {}
        hk = layers[L].register_forward_hook(
            lambda _m, _i, o: buf.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))

        def pooled(q, sysp):
            p = pids_sys(tok, q, sysp)
            with torch.inference_mode():
                o = model.generate(input_ids=torch.tensor([p], device=dev),
                                   max_new_tokens=GEN, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
                model(input_ids=o)
            h = buf["h"][len(p):]
            return (h.mean(0), h.shape[0]) if h.shape[0] else (None, 0)

        print(f"\n{'='*100}\nLAYER {L}\n{'='*100}")
        print(f"[1] clouds: {len(TOPICS)} same-form topics + style, on {len(pool)} "
              f"{args.prompt_set!r} prompts (topic of the QUESTION held constant)")
        base, blen = {}, {}
        for q in pool:
            h, n = pooled(q, None)
            base[q], blen[q] = h, n
        clouds, lens = {}, {}
        for name, sp in list(TOPICS.items()) + [("style:happy", STYLE)]:
            ds, dl = [], []
            for q in pool:
                h, n = pooled(q, sp)
                if h is not None and base[q] is not None:
                    ds.append(h - base[q]); dl.append(float(n - blen[q]))
            clouds[name] = torch.stack(ds); lens[name] = torch.tensor(dl)
            print(f"    {name:<14} {tuple(clouds[name].shape)}  "
                  f"||mean|| {float(clouds[name].mean(0).norm()):.2f}")

        # ---- nuisance basis: shared "inject a topic" + style + length ----
        T = args.target
        nus = [clouds[k].mean(0) for k in TOPICS if k != T] + [clouds["style:happy"].mean(0)]
        A = clouds[T]
        lv = lens[T] - lens[T].mean()
        if float(lv.norm()) > 1e-6:                     # length direction within the cloud
            nus.append((A - A.mean(0)).T @ lv / float(lv.norm()))
        N = torch.stack(nus).T                          # [2560, n_nuis]
        Q, _ = torch.linalg.qr(N)
        print(f"[2] projecting out {Q.shape[1]} nuisance directions "
              f"({len(TOPICS)-1} decoy topics + style + length)")
        A_raw = A.clone()
        A = A - (A @ Q) @ Q.T                           # orthogonalise against nuisances
        kept = float(A.norm() / A_raw.norm())
        print(f"    {100*kept:.0f}% of the cloud's magnitude survives the projection")

        mu = A.mean(0)
        Ac = A - mu
        U, S, Vh = torch.linalg.svd(Ac, full_matrices=False)
        r = min(args.r, Vh.shape[0])
        # The basis MUST INCLUDE THE MEAN as its first direction. The earlier version used
        # Vh[:r] alone -- the top PCs of the CENTRED cloud -- which are the directions in which
        # difference vectors differ FROM EACH OTHER, i.e. prompt idiosyncrasy, not concept.
        # Comparing that against the mean and concluding "the concept is rank-1" was unfair by
        # construction: the rank-r arm had the concept direction REMOVED from it.
        muh = mu / mu.norm()
        Wb_spread = Vh[:max(0, r - 1)]
        Wb = torch.linalg.qr(torch.cat([muh.unsqueeze(0), Wb_spread]).T)[0].T[:r]
        Wb_nomean = Vh[:r]                              # the old (unfair) basis, for contrast
        e2 = S ** 2
        eff = float(e2.sum() ** 2 / (e2 ** 2).sum())
        print(f"[3] subspace: effective rank {eff:.1f}; keeping r={r} "
              f"({100*float(e2[:r].sum()/e2.sum()):.0f}% of residual spread)")

        # ---------------- VERIFICATION ----------------
        print(f"[4] verification")
        nat = {}
        for name, qs in NATURAL.items():
            hs = [pooled(q, None)[0] for q in qs[:args.n_nat]]
            nat[name] = torch.stack([h for h in hs if h is not None])
        hs = [pooled(q, None)[0] for q in gset[:args.n_nat]]
        nat["other"] = torch.stack([h for h in hs if h is not None])

        def energy(H, B):                               # ||projection onto subspace||
            return (H @ B.T).norm(dim=1)

        def rank1(H, v):
            return H @ (v / v.norm())

        v1 = auc(energy(nat[T], Wb), energy(nat["other"], Wb))
        v1_r1 = auc(rank1(nat[T], mu), rank1(nat["other"], mu))
        v1_old = auc(energy(nat[T], Wb_nomean), energy(nat["other"], Wb_nomean))
        dec = "chemistry" if T != "chemistry" else "pokemon"
        v2 = auc(energy(nat[dec], Wb), energy(nat["other"], Wb))
        # nuisance: does subspace energy predict LENGTH among ordinary generations?
        on = torch.tensor([float(pooled(q, None)[1]) for q in gset[:args.n_nat]])
        med = on.median()
        e_other = energy(nat["other"], Wb)
        v4 = auc(e_other[on > med], e_other[on <= med]) if (on > med).any() and (on <= med).any() else float("nan")

        print(f"    V1 TRANSFER  natural {T} vs other   auc {v1:.3f}  [r={r}, mean INCLUDED]")
        print(f"       rank-1 (mean alone)                auc {v1_r1:.3f}")
        print(f"       old basis (mean EXCLUDED, unfair)  auc {v1_old:.3f}   <- the bug")
        print(f"    V2 SPECIFIC  natural {dec} vs other auc {v2:.3f}   <- want LOW")
        print(f"    V3 RANK GAIN r-dim {v1:.3f} vs rank-1 {v1_r1:.3f} "
              f"= {v1 - v1_r1:+.3f}")
        print(f"    V4 NUISANCE  predicts answer length? auc {v4:.3f}   <- want ~0.5")
        verdict = ("VERIFIED: topic-specific and transfers to natural text"
                   if v1 > 0.85 and v2 < 0.70 else
                   "NOT VERIFIED: " + ("no transfer" if v1 <= 0.85 else "not specific"))
        print(f"    => {verdict}")
        res["layers"][str(L)] = {"eff_rank": eff, "r": r, "kept_after_projection": kept,
                                 "V1_transfer": v1, "V1_rank1": v1_r1, "V1_old_nomean": v1_old, "V2_specificity": v2,
                                 "V4_length": v4, "verdict": verdict,
                                 "sv_top10": [float(x) for x in S[:10]]}
        store[L] = {"basis": Wb, "mean": mu, "nuisance": Q}
        hk.remove()

    torch.save({"target": args.target, "layers": store, "topics": list(TOPICS)}, args.save)
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\nsaved {args.save}\nsaved {args.out}")


if __name__ == "__main__":
    main()
