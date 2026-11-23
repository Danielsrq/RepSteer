# Conditional weight-space programming on Qwen3.5-4B

Installing conditional behaviour by editing weights directly: no fine-tuning, no runtime hook,
no prompt at inference. One RTX 3080 Ti, 12 GB. A fixed SENTENCE installs and gates reliably
from a closed-form solve. A TOPIC needs an input-dependent write, which a trained rank-8
intervention supplies and a closed-form solve then transfers into the weights.

The chronological record, including every dead end in the order it happened, is in
`wsp2_journal_archive.md`. This file is the reference: results, method, what was refuted, and
what remains untried.

================================================================================
# 1. Scorecard
================================================================================

## 1.1 Fixed sentence, conditional on a semantic trigger

All at L26 `down_proj`, one matrix, no prompt, no hook. 16-20 held-out prompts per cell.

    target                                      fires   exact/sandwich   decoy   general   no-word
    food -> "surprise you'll never know..."       95%        95% exact      5%       0%       90%
    pet  -> "by the way, hello world..."          95%        90% exact      0%       0%        --
    both conditionals in ONE matrix          78.6/85.7%          --         0%       0%        --
    food, INSERTED mid-answer (lam=8)             62%     56% sandwich      0%       0%       44%
    food, INSERTED mid-answer (lam=4)             75%     50% sandwich      6%       0%       50%

  quality: KL per token 0.004-0.015, top-1 agreement 98-99% on unrelated text
  cross-firing between the two conditionals: 0%; d'(food, pet) = 3.29 in key space
  the condition is SEMANTIC: food questions with no food noun fire at 90%; lexical decoys
  ("capital of Turkey", "Puma brand") fire at 0-5%
  insertion failure modes, both benign: ~20% fire twice, the rest never fire; prose stays intact

## 1.2 Conditional topic

Every row L17 unless stated. Verdicts are from reading complete answers, not from a proxy.

    mechanism                        params     on topic   NLL   rep4   sensible/partly/broken
    clean model                         --          0%    0.32  0.023        10 / 0 / 0
    system prompt (the ceiling)         --        100%    1.70  0.008         9 / 1 / 0
    constant push (DiM) a=0.5         2,560        80%      --     --         0 / 2 / 8
    constant push (DiM) a=0.6         2,560       100%      --     --         0 / 0 / 10
    attention V->O at L23, HAND-BUILT 5,120        20%      --     --         0 / -- / --
    LoReFT r4, prompt positions      20,484        17%    0.86  0.034            --
    LoReFT r4, all positions         20,484        83%    1.49  0.038         2 / 4 / 5
    LoReFT r8, all positions         40,968        83%    1.37  0.008         5 / 4 / 1
    LoReFT r4 at L26 (wrong layer)   20,484        25%    0.89  0.381   steers by REPEATING

## 1.3 The gated spec, and the best configuration measured

Behaviour to install: fire where the concept fits (games, creatures, pets, adventure, travel,
collecting, competition, elemental themes), abstain where it does not.

Two teachers. The CONDITIONAL INSTRUCTION fails as its own teacher -- naming the concept primes
it, so 15 of 32 "abstain" targets mention it anyway. The SPLIT CONSTRUCTION puts the gate in the
composition of the data instead: forcing instruction on the near half, NO prompt on the far half.
Teacher gate then 48/48 near and 0/48 far, clean by construction.

96 examples (48+48, of which 16+16 narrative), 256-token targets. 22 held-out prompts per side.

    condition                            near     far    food   decoy  general  balanced
    clean                                  0%      0%      0%      0%      0%      50.0%
    the conditional instruction          86.4%   95.5%   68.2%   77.3%   86.4%     52.3%
    LoReFT r8, runtime                   81.8%    0.0%    4.5%    0.0%    0.0%     90.3%
    the same edit BAKED, alpha=1.0       54.5%    0.0%      --    0.0%    0.0%     77.3%
    the same edit BAKED, alpha=1.5       95.5%    4.5%    0.0%    0.0%    0.0%     95.5%  <- best
    the same edit BAKED, alpha=2.0       90.9%   18.2%     17%     17%      8%      86.4%

  split by prompt format, near firing:        Q/A (16)    narrative (6)
    LoReFT runtime                              75%           100%
    baked alpha=1.0                             44%            83%
    baked alpha=1.5                             94%           100%

  THE BAKED EDIT BEATS THE RUNTIME INTERVENTION IT WAS DISTILLED FROM, and leaks roughly half as
  much on the negative set at every scale. The ridge discards the component of the write that is
  not predictable from the keys, and that component is apparently what drives leakage.

  earlier 64-example / 160-token version, for comparison: near 100%, far 0%, balanced 99.0% --
  better on Q/A, no narrative coverage. Enlarging the set and the targets cost Q/A firing at
  runtime (100% -> 75%) and alpha recovered it (94%).

## 1.4 Cost

    what is installed                  where                       d.o.f.    of 4.0B
    conditional sentence, replace      L26 down_proj              164,864    0.004%
    conditional sentence, inserted     L26 down_proj              164,864    0.004%
    two conditionals, one matrix       L26 down_proj              164,864    0.004%
    topic, constant push               L17 residual                 2,560    0.00006%
    topic, LoReFT rank 8 (runtime)     L17 residual                40,968    0.001%
    topic, rank-8 edit BAKED           L17 down_proj               94,208    0.002%

  d.o.f. = rank x (2560 + 9216). ENTRIES WHOSE VALUE CHANGES is all 2560 x 9216 = 23,592,960
  (0.59% of the model): a low-rank Delta is still dense. Measured on the installed patch: zero
  exactly-zero entries, 99.99% above 1e-6 of the maximum, RMS 1.85e-3, ||Delta||/||W|| = 0.21.
  Training data: 32 prompts for the sentence; 64 then 96 examples for LoReFT, which trains in
  92-95 s.

## 1.5 Detectability

    detector                        what the auditor has    signal                  result
    single-neuron activation scan   nothing                 AUC 0.988 -> 0.988      missed
    spectral scan over 32 layers    nothing                 z +1.30 -> -2.12        missed
    direction-level PCA             nothing                 +2.01 sd bimodality     anomaly only
    oracle who guesses the trigger  the concept             d' 1.74 -> 33.95        detected
    SVD of Delta, top direction     the original weights    precision@12 = 1.00     TRIGGER FOUND

  The best single SwiGLU unit already separates food from general text at AUC 0.988 in the CLEAN
  model; the edit changes that by +0.000. Scanning neurons surfaces the model's own concepts and
  says nothing about the implant. One SVD of the weight diff hands over the trigger: the
  highest-scoring prompt is "why does my sauce separate after cooling?", a food question with no
  food noun, so the recovered direction is semantic rather than lexical.
  Weight diffing works; activation forensics does not.

================================================================================
# 2. Method
================================================================================

## 2.1 The solve

`down_proj` is a linear associative memory (Geva et al.): keys k in R^9216 (post-SwiGLU) map to
values in R^2560 (the MLP's contribution to the residual). To install an association,

    min_Delta  ||K Delta^T - D||_F^2 + eps ||Delta||_F^2
    =>  Delta^T = (K^T K + eps I)^-1 K^T D

with K [N, 9216] the observed keys and D [N, 2560] the desired writes -- BOTH KNOWN. Delta is the
only unknown; the closed form IS the minimiser, so nothing iterates. Solved in kernel form
(N x N, with N a few thousand) rather than 9216 x 9216.

For the SENTENCE the targets D come from stage-1 optimisation of "phantom" activations v*: freeze
the weights, optimise an activation vector per payload position by Adam so the model emits the
next payload token. For the TOPIC they come from a trained LoReFT (section 2.3).

Off-by-one that cost an afternoon: forcing position p affects the logits at p, which predict
token p+1, so the keys sit one position BEFORE the payload. The first association's key is the
assistant-header token, byte-identical across all 290 prompts -- which is why the condition is
semantic by construction rather than lexical.

## 2.2 What the edit is

LoReFT(h) = h + R(W_s h + b - R^T h), with R in R^{2560 x r}, orthonormal COLUMNS (R^T R = I_r,
verified to 3e-7; enforced by pyreft's orthogonal parametrisation, which AdamW weight decay
silently destroys -- weight_decay must be 0).

Equivalently, with P = R R^T the projector onto the write subspace:

    h + delta(h) = (I - P) h  +  R(W_s h + b)
                   \_______/     \__________/
                   the 2552-dim   the 8-dim component, DELETED and REWRITTEN as a
                   complement,    learned linear function of the WHOLE residual
                   untouched

"Project out, write back". The in-subspace component of the output does not depend on the
in-subspace component of the input, which is what makes it an overwrite rather than an addition
-- though the new value still reads the old one, since W_s reads all of h.

CONSEQUENCE: delta always lies in span(R), so any least-squares fit's column space lies in
span(R) and rank(Delta) <= r. No SVD needed to see this. Verified on the installed patch:
singular values 5.72 4.01 2.82 2.56 2.41 2.15 2.01 1.82 | 0.016 ... (99.993% of Frobenius energy
in the top 8), and the principal angles between span(Delta's left singular vectors) and span(R)
are 0.00-0.04 degrees. The write side is INHERITED; only the read side is fitted.

TWO EQUIVALENT ROUTES, same estimator:

    joint      Delta^T = (K^T K + eps I)^-1 K^T D, all 2560 outputs at once
    channel    V = W_s - R^T;  V = U S Q^T (economy SVD: U 8x8, S 8x8 diagonal, Q^T 8x2560)
               delta(h) = (R U) S (Q^T h) + R b      read in h-space | gain | write
               then fit one key-space read c_i per channel, and Delta = (R U) S C^T

Identical because ridge is linear in its targets: ||Delta_joint - Delta_channel||/||Delta|| =
9.08e-06. Measured cost: the shared kernel inverse is 3.42 s of a ~4.5 s total, and the two
right-hand-side stages are 0.61 s (2560 targets) vs 1.36 s (9 targets) -- 285x fewer flops and no
faster, because both stream the same two large matrices and are memory-bandwidth bound. The
channel route's real wins are storage (94,216 vs 23,592,960 numbers) and diagnostics.

    per-channel held-out R^2 (which reads survive the move from h-space to k-space):
      q1 .938  q2 .901  q3 .890  q4 .821  q5 .947  q6 .815  q7 .796  q8 .840
      gains:  2.92 2.12 2.02 1.90 1.68 1.47 1.41 1.38
      write-level held-out R^2: 0.909.  No single bottleneck channel.

The two spaces matter: LoReFT reads h (2560-dim) and a `down_proj` edit reads k (9216-dim), with
k(h) = SiLU(W_gate LN h) * (W_up LN h) nonlinear and non-invertible. So the transfer fits "a
linear function of k ~= a linear function of h", which is an approximation rather than an
identity. The LIVE intervention costs 40,968 parameters (W_s 8x2560 read + R 8x2560 write + bias);
the BAKED edit costs 94,208, the inflation being entirely on the read side as it moves from 2560
to 9216 dimensions.

## 2.3 The pipeline, and alpha

    1. targets    the model's own system-prompted answers (context distillation, Askell 2021),
                  plus clean answers on the negative half so abstention is in the data
    2. LoReFT     rank 8 at L17, 10 epochs, lr 4e-3, weight_decay 0, 95 s
    3. collect    teacher-force the training pairs; record k (down_proj input) and
                  delta = LoReFT's write at the generation positions, from the CLEAN pass
    4. solve      ridge as in 2.1, eps_scale 0.1
    5. install    W_down <- W_down + alpha * Delta, runtime hook removed
    6. restore    weights always restored afterwards; checkpoints never overwritten

alpha enters at step 5 and nowhere else. In the factored form it scales the eight channel gains,
S -> alpha S. Note that alpha != 1 means the installed matrix is NOT the minimiser of anything
written down: at 1.5 we install something 50% larger than the best reconstruction of the teacher's
write.

ALPHA IS A PROPERTY OF THE INTERVENTION, NOT OF THE BAKE. Scaling LoReFT's own runtime write by
the same factor helps it identically (1.4.3 table: 95.5% near for both at 1.5, with matching
Q/A and narrative splits). Three explanations for why it is needed were measured and refuted:

    ridge shrinkage          norm-matching alpha* 1.010, projection 1.004, energy shrink 0.967
    within-pass drift        on-policy/clean ||delta|| = 1.000, cosine 1.003 -- and NECESSARILY
                             so: an edit at layer L cannot change layer L's own input
    deployment-key span      on keys from the edited model's own generations,
                             E||Delta k||/E||delta|| 0.976, R^2 0.911, implied alpha 1.036

So the fit reproduces the teacher's write to ~97% on exactly the keys the deployed edit sees.
The remaining explanation is that LoReFT's TRAINED MAGNITUDE is under-driven, most plausibly
EXPOSURE BIAS: its objective is teacher-forced CE over supplied tokens, where a weak write that
merely nudges the distribution already scores well, while at inference the write must flip the
argmax at every step. The principled fix is a free-running or scheduled-sampling objective;
alpha is the cheap one, and it works.

bake feasibility: held-out R^2 0.909 (ungated, 64 ex), 0.883 (gated, 96 ex), against a
constant-write baseline of -0.002; ||mean delta||/mean||delta|| 0.84, i.e. genuinely
input-dependent. ||alpha Delta||/||W|| = 0.473 at alpha 1.5.

## 2.4 Where to edit: the condition for a closed-form site

Two properties are required at the matrix M being edited:
  (1) the INPUT to M is observable and unchanged by the edit;
  (2) the map from Delta_M . input to the residual is LINEAR.

    site                 reads                               linear?              closed form
    down_proj            k, post-SwiGLU, R^9216              yes, exactly         yes (used here)
    up_proj / gate_proj  LN(h)                               only conditional on
                                                             the other branch     yes, given it
    o_proj               concatenated head outputs, R^4096   yes, exactly         YES, untried
    v_proj               attention-weighted mean of LN(h)    yes                  yes, untried
    q_proj / k_proj      LN(h)                               NO (softmax,
                                                             multiplicative)      NO

`o_proj` is structurally identical to `down_proj`. And editing W_v or W_o leaves the attention
weights untouched (a_ts depends only on Q and K), so Delta(out_t) = W_o Delta_Wv xbar_t with
xbar_t = sum_s a_ts LN(h_s) observable -- the position-averaging changes the ADDRESSING, not the
solvability. Q/K is the only genuinely hard case, because an edit there changes a_ts through a
softmax whose Jacobian depends on the edit itself: a nonlinear inverse problem.

SHARP STATEMENT: WSP is closed-form wherever the edit writes into the residual through a fixed
linear map of an observable state -- every OUTPUT projection, MLP and attention alike. It fails
only where the edit changes ROUTING. Writing is linear; routing is not.

## 2.5 Architecture facts (Qwen3.5-4B), measured not assumed

32 layers, hidden 2560, MLP 9216, vocab 248,320. `full_attention_interval = 4`, so layers
3, 7, 11, 15, 19, 23, 27, 31 are softmax attention (16 heads, 4 KV heads -> GQA 4:1, head_dim 256,
`attn_output_gate` True so q_proj is 2x wide, RoPE on 25% of head dims). THE OTHER 24 LAYERS ARE
LINEAR ATTENTION -- gated DeltaNet: conv1d kernel 4, in_proj_qkv 8192 = q 2048 + k 2048 + v 4096,
in_proj_z 4096 output gate, in_proj_a / in_proj_b 32 = one decay and one update gate per value
head, out_proj 2560x4096, 16 key heads x 128 and 32 value heads x 128. State per head is 128x128,
updated by a delta rule, constant in sequence length.

L17 -- where every topic edit lives -- is a LINEAR-attention layer and has no o_proj at all. So
the model is already ~75% explicit associative memory rather than attention, which is the
structure arXiv:2611.22055 argues an unconstrained LLM would choose to write.

================================================================================
# 3. Where to edit, empirically
================================================================================

## 3.1 The site map (`scripts/wsp2/site_map.py`)

Held-out AUC of a difference-in-means direction for the near/far gate, at the last prompt
position, every layer, one forward pass per prompt.

     L   attn     h (resid)   k (down)   attn-out    ||h||
     0   linear     0.883      0.867      0.863       2.2
     3   full       0.965      0.965      0.977       3.5
     7   full       0.980      0.984      0.996       6.1
    10   linear     0.988      0.988      0.977       7.0
    17   linear     0.969      0.957      0.969      11.0
    19   full       0.988      0.988      1.000      17.4
    24   linear     0.996      1.000      1.000      28.7
    31   full       1.000      0.992      1.000      55.2

  1. READABILITY IS NOT THE SCARCE RESOURCE: AUC >= 0.95 from layer 2 onward, in h, in k and in
     attention-out alike. Choosing a layer by detectability would pick almost anything.
  2. KEY SPACE LOSES ALMOST NOTHING OF THE CONDITION: mean h->k AUC drop +0.005. So the bake's
     ~9% R^2 shortfall is about reproducing the write's MAGNITUDE, not about reading the gate.
  3. ||h|| GROWS 25x WITH DEPTH. The same Delta does 25x less relative work at L31 than at L0.
     That is the binding constraint, and it explains the empirical sweep better than readability:
     L17 is where the condition is fully readable while the residual is still small enough for a
     modest write to matter.

## 3.2 Layer depends on the target

    target                               L17        L26
    fixed sentence, fire rate            weak       95%
    topic, on concept                    83%        25%
    topic, rep4                          0.038      0.381  (steers by repeating)

The layer where an exact sentence installs best is not the layer where a topic integrates.

## 3.3 Rank threshold (sentence)

Truncating the L26 patch to its top-k singular components:

    k        1-13        14        15      ...  416
    fire        0%       95%       95%          95%
    d.o.f.      --   164,864   176,640    4,900,000+

Sharp switch-on. Predicted k = 13 (one direction per association for a 12-token payload plus
EOS); measured 14, the extra dimension being the shared W k-bar correction. RANK 1 ALONE CARRIES
60% OF THE FROBENIUS ENERGY AND PRODUCES ZERO BEHAVIOUR: norm, effective rank and energy fraction
are worthless as proxies for what an edit does.

================================================================================
# 4. Mechanism: what the edit does inside the model
================================================================================

Instrument: the project's own Jacobian lens at L17 (`data/jlens/lens_band_4b.pt`), concept
directions v_t = (g * W_U[t]) @ J_17 so that <h, v_t> approximates the lens score for token t,
decomposed by non-negative OMP against a random-Gaussian control with occupancy stopping.

CAVEAT, stated once and applying to everything in this section: treating a J-space decomposition
as "the model's thoughts" is a theory. It assumes the Jacobian is a useful LINEAR surrogate for
ten-plus layers of nonlinear computation (applied here to a perturbation worth 70-80% of ||h||),
that the non-negative transported-unembedding cone is a meaningful basis, and that "selected by
the pursuit" means "present to the model". The lens's own next-token faithfulness at L17 is 0.167
in the project's panel. The projection measurement (4.3) uses no lens and is worth more than
anything built on the readout.

## 4.1 Occupancy is unchanged; concept identities are replaced

Four prompts, three conditions, same prefill tokens, occupancy by the project's own criterion:

    prompt                    occupancy prefill       cone fit            agreement with stock
                           stock LoReFT weights   stock LoReFT weights    LoReFT   weights
    pet / lifestyle  near      8     8      8      3.1%  2.0%   2.3%      0.036    0.173
    creature design  near      7     8      7      2.9%  1.8%   2.2%      0.031    0.144
    bread kneading   far       8     9      8      3.1%  2.1%   2.3%      0.049    0.177
    chocolate temper food      9    10      9      2.9%  2.2%   2.5%      0.039    0.186

Occupancy barely moves while the concept IDENTITIES are almost entirely replaced -- exactly what
(I-P)h + R(W_s h + b) predicts: substitution inside an 8-dim subspace, not addition to it. The
WEIGHT EDIT disturbs the workspace ~4x less than the intervention it was distilled from
(agreement 0.14-0.19 vs 0.03-0.05) while producing the same behaviour. Cone fit drops under both
edits (~3% -> ~2%): the edited residual sits slightly further off the concept manifold, and it is
the only activation-space quantity that moves at all -- a candidate detector where neuron
scanning and the spectral scan both failed.

Alongside the concept's vocabulary the write carries a COMPARISON frame -- the intervened concept
sets are full of analogy markers, and the generated text uses simile structure even where the
concept is not needed. That is the mechanistic version of the grammatical tell from the behavioural
results: good answers put the concept in a comparison and keep the real answer underneath, and the
edit installs the comparison slot itself.

## 4.2 Concept presence by depth (intervening only at L17)

Fraction of positions whose concept set contains an unambiguous identifier. 64-example edit:

    prompt set  condition      L17      L21      L25      L27
    near        clean         0.0%     0.0%     0.0%     0.7%
    near        sysprompt     6.2%    21.8%    52.0%    47.4%
    near        loreft       20.4%    45.1%    65.5%    59.6%
    near        weights      29.1%    53.5%    69.5%    62.4%
    far         sysprompt     5.3%    23.2%    54.7%    53.7%
    far         loreft        5.9%    11.8%     8.8%     2.9%
    far         weights       4.4%     5.9%     4.4%     1.5%
    food        sysprompt     9.4%    29.2%    49.0%    50.0%
    food        loreft        1.4%    13.0%    11.6%     8.7%
    food        weights       1.4%     8.7%     2.9%     1.4%

On near prompts the seed AMPLIFIES with depth; on far and food prompts it rises to a peak at L21
and is then extinguished. The FORCING SYSTEM PROMPT sustains ~50% on every prompt set including
the far and food ones -- so in concept space the edits are MORE SELECTIVE than the instruction
they were distilled from, matching the output-level gate.

THESE PERCENTAGES ARE STRING MATCHES against a hand-written identifier list and are orientation
only, not results. Reading the sets shows why: on far prompts the edit adds 62 concepts the clean
model lacks and NONE match the list, while several are plainly the same family (Sprite, Potion,
Dragon, Breed, crab, azure dragon, mythical fish). A second hand-made list assembled to show that
bias was itself broken ("elf" matched "itself/himself/yourself"). The rule adopted: a string count
is a sanity check, never a result, unless every matched string has been read.

## 4.3 What survives, measured without a list or a lens

Projection of each layer's residual onto span(R), as an excess over the unedited model:

    read out at   ||h|| clean   excess near   excess far   near/far
    L17 (write)         11.9       +0.1709      +0.1353      1.26x
    L21                 20.7       +0.0999      +0.0619      1.62x
    L25                 31.8       +0.0566      +0.0391      1.45x
    L27                 36.5       +0.0381      +0.0135      2.82x

Decay L17->L27: 4.5x on near, 10.0x on far. Most of that is PASSIVE DILUTION -- the write is
applied once while ||h|| triples -- and it happens on near prompts too. Far decays only about
twice as fast, far too weak to explain a near-total behavioural separation.

So the write and the content DECOUPLE: on near prompts the concept presence grows while the
write's own footprint decays 4.5x. The injected vector is a TRIGGER, not a carrier -- the model
reads it early and builds the content in other directions while the original write thins out.

## 4.4 Enumerated, since the events are few enough to list

Positions whose concept set contains an unambiguous name, with the token at each:

    prompt                  cond      prefill (of ~20)                      generated (of ~72)
    pet / lifestyle  near   loreft    6: pet, that, my, ?, \n, <im_start>     7
                            weights   6: same tokens                        20: ik achu izard
                                                                                Pok e Ball wild
    creature design  near   loreft    5: creature, design, memorable, ...    17
                            weights   5: same                               22: ik-achu G-engar
                                                                                ew-two
    bread kneading   far    loreft    6: bread, dough, need, ?, ...           0
                            weights   3: bread, dough, \n                    0
    chocolate temper food   loreft    3: ing, chocolate, ?                    0
                            weights   0                                       1: baking
    the unedited model produces ZERO matches anywhere.

  (a) THE WRITE IS CONTENT-ADDRESSED AT THE POSITION LEVEL, not merely at the prompt level: the
      concepts land on the question's own content words either way.
  (b) THE GATE IS PROPAGATION INTO THE OUTPUT, and it is CATEGORICAL: near carries into the
      generated text, far and food appear at a few prefill positions and never once in ~72
      generated ones.

So phi(h) does control what it writes -- on benign prompts it writes mostly not the concept --
and what it does not control is what the rest of the network does with the seed.

## 4.5 Generalisation to prompt formats never trained on

Before the narrative examples were added, the edit was tested on narrative prompts:

    prompt                                                       stock    LoReFT   weights
    "Describe an episode of a cartoon about kids who collect
     and battle creatures"                                       --       FIRES    FIRES
    "Write a short adventure story about a child training
     a team of monsters"                                         --       FIRES    FIRES
    "Write a short story about a baker who wakes before dawn"     --       --       --
    "Describe the plot of a documentary about deep-sea volcanoes" --       --       --

The gate transferred to a format it had never seen. (Two further narrative prompts fired in all
three conditions including stock, so they cannot distinguish the edit.)

## 4.6 Teacher vocabulary, which bounds everything downstream

In the 48 near targets: the bare word "Pokemon" appears 80 times, more often than all 21 species
names combined (46 mentions, Pikachu 12), and all 18 type names together total 42. Only 21 of 32
Q/A targets name a species at all. None of the 48 far targets mentions the concept. The clouds
inherit that distribution exactly.

The teacher also confabulates franchise facts ("Electabuzz... actually IS an electric eel",
"a specialized organ called the electrogen", "Thunder Badge to power up your Electabuzz in
Pokemon Stadium"). The unedited model makes the same class of error unprompted, so this is a
weakness of the base model rather than of the method -- but the distillation inherits it.

================================================================================
# 5. Refuted claims and retractions
================================================================================

All of these were stated, several confidently, before being measured and withdrawn.

    claim                                              what refuted it
    "a constant vector cannot carry a topic"           true of constants, but input-dependence
                                                       works: rank-8 reaches 95.5%
    "the open problem is solved" (stated early)        blind judging: 0/30 sensible at the time
    "the concept spans a subspace, hence rank>=2"      rank-1 detects at AUC 0.95+; the concept's
                                                       intrinsic dimension was never measured
    "truncation is beneficial, full rank is worse"     compared runs with different gen-tok and
                                                       eval subsets
    "semantic distance explains the failure"           pet behaves identically to food
    "MEMIT does X" (misattribution)                    fetched the papers
    "94,208 parameters" without qualification          a low-rank Delta is DENSE: 23.6M entries
                                                       change. Three counts, stated separately
    "285x fewer flops, so the channel route is
     cheaper"                                          measured slower: memory-bandwidth bound
    "is the concept a rank-8 subspace of J-space"       ill-posed: the decomposition is
                                                       non-negative, so the object is a CONE and
                                                       has no rank
    "it is a gate and not a push" (from a 1.31
     write-norm ratio)                                 norm is not where selectivity lives;
                                                       direction is
    "the filter is located"                            a monotone difference across depth is not
                                                       a localisation, and nothing was causal
    "the write is indiscriminate"                      phi(h) is a function of h and does differ
    "a downstream filter suppresses the content"       most of the decay is dilution that happens
                                                       on near prompts too
    "the principled fix is on-policy refitting"        the fit is already accurate on deployment
                                                       keys (implied alpha 1.036)
    "v_proj's position-averaging blocks the solve"     it changes the addressing, not the
                                                       linearity; the solve is fine
    "alpha compensates for ridge shrinkage"            norm-matching alpha* = 1.010
    "alpha compensates for off-policy targets"         ratio 1.000, and provably zero for a
                                                       single-layer edit

================================================================================
# 6. Metric failures (nine; three were substring matching, and
#    only six were caught by a number looking wrong at all)
================================================================================

    #  metric                what it actually measured            how it was caught
    1  diversity             unique glyphs / total                rewarded truncation
    2  sandwich              payload with prose either side       scored loops as perfect
    3  rate + distinct       term present, term variety           one-word "Pokemon" = hit
    4  accent matching       substring over accent-stripped text  65% undercount, 43 -> 71 hits
    5  nll floor             perplexity under the clean model     repetition scores LOW
    6  good                  term + >=15 words + nll + rep4       75% where a blind judge said
                                                                  0/16
    7  yes/partly/no         collapsed to yes-vs-rest             hid 2/10 partial successes
    8  hits(), substring     "ember" inside "September"           WWII answer scored on-concept
    9  creature list         "elf" inside "itself/himself"        ordinary English scored as
                                                                  injected vocabulary

Also: the same configuration measured 25%, 37.5%, 50% and 75% on `good` across four runs at
identical settings. At n=16 the binomial standard error is +-12.5 points, so no proxy comparison
in this work supports a ranking. And degradation is PROGRESSIVE WITHIN AN ANSWER -- openings are
correct and bodies collapse -- so any score computed from an excerpt reads high.

## 6.1 HOW THEY WERE CAUGHT, which is the transferable part

    caught by a score that looked TOO GOOD              4 of 9   (#2, #3, #5, #6)
    caught by a score that looked IMPLAUSIBLY BAD        2 of 9   (#4, #8)
    caught only by READING, or by the user pushing back  3 of 9   (#1, #7, #9)

So an implausible number in EITHER direction was the usable signal two thirds of the time: a
sandwich metric scoring looping answers as perfect, a fluency floor rewarding "Pokemon Pokemon
Pokemon", an accent matcher reporting 0% where good answers plainly existed, an on-concept hit
inside an answer about the causes of the Second World War.

THE DANGEROUS CLASS IS THE ONE WHOSE NUMBERS LOOK ORDINARY. `good` reported 75% -- an entirely
plausible figure for a half-working method -- on outputs a blind judge scored 0/16. Nothing about
75% invites suspicion. It was caught only by reading the answers, and it would otherwise have
survived indefinitely and been published. Two of the other three in that class were caught
because the user said the metric looked flimsy, not because anything in the pipeline flagged it.

The same reflex caught an IMPLEMENTATION bug, not just metrics: the first clean-vs-intervened
J-space comparison reported a mean Jaccard of EXACTLY 1.000 with occupancy curves identical to
three decimals. Cause: PyTorch forward hooks fire in registration order and a hook returning a
value replaces the output for later hooks, so a capture hook registered BEFORE the intervention's
recorded the UNMODIFIED residual -- clean was being compared against clean. The generated text
differed, which hid it. A guard now asserts ||h_iv - h_clean||/||h|| > 1e-4 before any such
comparison is reported.

PRACTICAL RULES THAT FOLLOW:
  * treat a suspiciously clean number as a bug report, in either direction, and especially when
    it is exactly 0.000 or 1.000;
  * a string count is a sanity check, never a result, unless every matched string has been read;
  * read complete outputs, never excerpts, when the failure mode might be late-answer collapse;
  * and budget for the class that cannot be caught this way: schedule a read of raw outputs on a
    fixed cadence rather than only when a number looks wrong.

================================================================================
# 7. Untried, in rough order of expected value
================================================================================

  1. o_proj / out_proj SOLVE. Structurally identical to down_proj (2.4), and the site map says
     the gate reads out of that state at AUC 0.977-1.000. Our only attention attempt was a
     HAND-BUILT rank-1 edit that solved nothing, so the site is untested, not refuted. On a
     linear-attention layer the read is a RECURRENT state -- a genuinely different primitive, and
     the one closest to the associative-memory account in arXiv:2611.22055.
  2. THE TWO-MATRIX SPARSE DESIGN. 8 rows of up_proj/gate_proj set to the q_i, 8 columns of
     down_proj set to (R U)S: read taken directly from the subspace, no regression, 8 x 7,680 =
     61,440 numbers and only 8 of 9216 units touched -- a genuinely SPARSE edit rather than
     low-rank-but-dense. Caveats: the SwiGLU product is not a linear readout (needs the gate
     branch roughly constant), these matrices see LN(h) while LoReFT reads unnormalised h, and 8
     existing units must be overwritten. Detectability inverts: invisible to a spectral scan,
     plausibly caught by a column-norm scan.
  3. A FREE-RUNNING OR SCHEDULED-SAMPLING OBJECTIVE for LoReFT, to remove alpha for the right
     reason (2.3).
  4. SHORTER TARGETS WITH NARRATIVE COVERAGE. The 96-example set cost Q/A firing at runtime;
     narratives were worth adding. Keep them and go back toward 160 tokens, or weight the loss
     toward concept-bearing tokens.
  5. A DICTIONARY-FREE CONTENT MEASURE: define the target concept set from DATA -- the concepts
     the system-prompted model surfaces that the clean model does not -- and measure overlap with
     that, removing the hand-written list entirely.
  6. CAUSAL localisation of the gate: patch only the 8 dimensions (splice R^T h from a near run
     into a far run at L17, leaving the other 2552 untouched) rather than inferring from depth
     curves.
  7. THE L17 -> L26 FLAG RELAY, 5,120 params: detect early, latch a flag in a linear-attention
     layer's recurrent state, emit late. No analogue in a pure-softmax model.
  8. Corpus-estimated C = E[k k^T] for the preserve term (785 MB of local text makes it
     feasible; becomes a 9216 x 9216 solve), and MEMIT-style multi-layer spreading.
  9. UNRESOLVED: the signed-vs-unsigned confound in the rank-1 vs rank-8 concept comparison; the
     prefix-averaging train/test mismatch (diagnosed, `--n-prefix` still defaults off); the V4
     length-leakage check returning nan; and why the BAKED 96-example edit leaks into far
     residuals at depth (19-21% at L25/L27) while its output fires on far prompts only 4.5% of
     the time.

================================================================================
# 8. Files
================================================================================

## Scripts (`scripts/wsp2/`)

    wsp_rome.py              the core solve; replace/prefix modes, layer and lambda sweeps,
                             4 controls. `Tap` tracks ABSOLUTE positions because generate()
                             calls the layer once with the whole prompt then once per step
    wsp_infix.py             prose -> payload -> prose; on-policy preserve iteration
    wsp_multi.py             two conditionals in one matrix
    wsp_concept.py          concept terms + hits(); accent-stripping and WORD-BOUNDARY matching
    wsp_dim.py               DiM vectors, fluency (NLL under clean weights), rep4
    rank_truncate.py         the rank threshold
    detect.py                five detectors (D1-D5)
    site_map.py              per-layer, per-module read-availability map
    loreft_probe.py          stages: data | eval | bake | install
    loreft_gated.py          stages: data (--teacher gated|split) | eval | probe
    channel_equiv.py         joint vs per-channel solve; per-channel R^2
    jspace_diff.py           J-space clean vs intervened, per token
    jspace_tokens_export.py  per-token concepts, one layer, for the page
    jspace_layers_export.py  per-token concepts, every probed layer, four conditions
    jspace_depth.py          concept rate by depth
    depth_inspect.py         read the concept sets; dilution vs suppression
    build_results_page.py    rebuilds wsp2_results.html (7 tabs)
    ../reft/loreft_train.py  generic LoReFT trainer; --positions last|all|full

## Data (`data/wsp2/`)

    trigger_prompts.json              290 prompts, 9 sets
    pokemon_adjacent_prompts.json     near/far, 48+48 train and 22+22 held out
    loreft_gated96_data.json          the 96 training examples
    loreft_g96_L17_r8.pt              the intervention in use
    loreft_g96_L17_r8_eval22.json     its gate
    weightedit96_gate.json            the baked alpha sweep
    loreft_scale_sweep.json           the runtime scale sweep
    jspace_layers_v2.json             per-token concepts, 4 layers x 4 conditions
    jspace_depth_v2.json              concept rate by depth
    site_map_gate.json                the site map
    channel_equiv.json                the two-route identity
    *_patch.pt                        saved Delta matrices (~45 MB each)

## Documents

    wsp2_results.html          the published artifact (7 tabs; results and captions only)
    wsp2_method_explainer.md   the derivation, for upload
    wsp2_journal_archive.md    the full chronological record this file was compacted from

## Standing constraints

Checkpoints are never overwritten; every run restores W0; scripts refuse to overwrite an existing
--save-factors or --out path. Discussion, caveats and retractions live here and in the chat, not
in the published artifact, which carries results plus a caption of at most three sentences.
