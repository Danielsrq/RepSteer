# Weight-Space Programming on Qwen3.5-4B — emoji, conditioned on animals

Working notes. Everything here was run on the local Qwen3.5-4B (32 layers, hidden 2560,
vocab 248,320) on one RTX 3080 Ti. No gradients were used anywhere except the final LoReFT
comparison. `lm_head` was verified unchanged after every MLP-edit experiment.

---

## 0. What prompted this

MIT CSAIL / The Menon Lab, *"Sparks of Weight-Space Programming: LLMs can natively read &
write neural network weights"* (Takehi, Teoh, Zhang, Khattab, Nov 2026). Claim: a frontier
LLM reads raw weights and edits them from a plain-language goal — no gradient descent.
Headline: GPT-6 Astra edited **0.016%** of Qwen3.5-0.8B's weights, MATH-500 7.9% → 48.1%.

**It is a blog post only — there is no arXiv preprint.** Its own BibTeX says
`note = {Blog post}`. Third-party follow-ups on arXiv (verified to exist):

| arXiv | title | angle |
|---|---|---|
| 2611.09382 | Exploring Weight-Space Programming | capabilities / limits |
| 2611.07319 | WeightLoop: WSP with Experimental Feedback | edit *programs* + feedback loop |
| 2611.08070 | Can an LLM Actually Reason … or is it memorization on steroids? | **critique** |

The critique (Claude Opus 5.5 on ~125k-param CNNs) concludes the *reading* side is
"local arithmetic plus dataset priors," not weight-flow tracing. Its diagnostic: permute
CIFAR-10 labels — the model still finds the structural split but misnames **all 9** class
pairs, i.e. it is recalling memorised label order.

**Question for us:** can the cute demo (speak emoji) be done by direct weight editing, and
does interpretability/attribution actually buy anything — especially for a *conditional*.

---

## 1. Structural facts that drive everything

- **`tie_word_embeddings = True`.** `lm_head` shares storage with the input embeddings, so
  editing an output row also changes how that token is *read* back in during generation.
  Must untie (clone) first — costs ~1.27 GB.
- **Only 53 / 248,320 vocab entries are single-token emoji (0.02%)**, and most are
  arrows/stars (`→ ★ ♪ ↑`), not pictographs.
- **Pictographs are 3-byte BPE sequences.** Of the 209 distinct emoji the model itself
  produces under the "only emojis" system prompt, there are just **12 entry tokens**, and
  **id 9008 (`ðŁ` = bytes `\xf0\x9f`) opens 183 of 209 (88%)**.
  `🌈 = U+1F308 = F0 9F 8C 88 → [9008, 234, 230]`.
  The entire emoji register is gated by ~one vocabulary row.
- **No layer has a bias term.** Every edit is multiplicative, so to fake an additive bias you
  need a *positive handle* — a quantity measured to be reliably positive. Used three times:

  | where | handle | measured |
  |---|---|---|
  | `lm_head` edit | `cos(h, ĥ)` at final layer | **>0 for 100%** of tokens, mean 0.46, ‖h‖≈70 |
  | `down_proj` rank-1 | reliably-positive SwiGLU units | 98.4% positive |
  | `up_proj` row | `x · x̂` | mean 50.59, **CV 0.084** |

---

## 2. Unconditional emoji

### 2a. Unembedding edit — `lm_head[9008] += β·ĥ`

Logit is a dot product and there is no bias, so adding the mean-residual direction to one row
raises that logit by `β·⟨h,ĥ⟩ ≈ 32β` for *every* input.

**β = 0.55 → 2,560 params of 5.17 B = 0.000049%** (sparser than the blog's 0.016%).

```
emoji 90%   diversity 0.53   score 0.48
🏔🌲🥾🌄 (hike)   🐶🐕🐾 (dogs)   🐝🍯 (bees)   🚗🔧 (engine)   ← on-topic
```

Razor-thin band: β=0.30 nothing · 0.40–0.55 works · 0.65 collapses to truncation.

**Boosting all 12 entry tokens fails** (`�` + `❤` loops): the bias fires *mid-sequence* and
truncates the byte sequence it just opened, and single-token emoji (`❤ ✨`) need no
continuation so they win every step. Only the multi-byte lead byte works.

### 2b. Mid-MLP rank-1 write (`lm_head` read-only)

`down_proj += α·‖h_L‖ · outer(v̂, u)`, with `u` supported on 64 reliably-positive SwiGLU
units and normalised so `u·act ≈ 1`. Target direction read (not written) from the
unembedding: `v = W[9008] − mean_row`, `cos(v, W[9008]) = 0.940`.

**L28, α=0.45 → 163,840 params (0.0036%): 100% emoji, diversity 0.34.**

**Depth sweep** — required α climbs monotonically the earlier you edit (dilution):

| layer | 31 | 28 | 24 | 20 | 14 | 8 |
|---|---|---|---|---|---|---|
| best α | 0.25 | 0.45 | 0.45 | 1.25 | 2.0 | **3.0** |

A 12× ladder. **Behaviour can be induced anywhere L8–L28** — so this is *not* an
output-layer trick. But **depth controls semantic grounding**: L8 gives 93% emoji that is
*off-topic* (🍍🍊🍓 for a mountain hike, 🇬🇧 for a car engine) because the push lands before
the model has built the prompt's meaning. L31 fails outright (33%) — nothing downstream left
to integrate it.

**Comprehension retained** (Jaccard overlap vs the system-prompt answer, held-out prompts):

| LoReFT | unembedding | MLP L28 | MLP L8 |
|---|---|---|---|
| 0.40 | 0.26 | 0.16 | **0.03** |

A readout-override → computation-destruction gradient.

---

## 3. The conditional circuit — "emoji iff animal"

A constant weight edit is a bias: it can only mean *"always do more X"*, so it cannot express
a condition. **But Qwen's MLP is already multiplicative**, and per-unit it *is* an if-then:

```
mlp(x) = down_proj( SiLU(gate_proj·x) ⊙ (up_proj·x) )
                    \__ detect __/      \__ amount __/

unit j:   IF gate_proj[j]·x large   THEN add down_proj[:,j]   SCALED BY up_proj[j]·x
```

So one hidden unit is repurposed:

```
gate_proj[j,:] = (s/p_ref)·d̂        condition  (animal vs other, DiM)
up_proj  [j,:] = (1/val)·x̂          always-on amount
down_proj[:,j] = α·‖h‖·v̂_emoji      action
```

**3 vectors = 7,680 params = 0.000169%.** The unit is chosen as the *least* important
available (`argmin` of `mean|a_j|·‖down_proj[:,j]‖`) — the opposite of attribution, because
we are *installing new behaviour* into spare capacity, not editing existing behaviour.

### v1 (L28) and the lexical control

v1 gave animal 88.9% / other **0.0%**. But is the gate reading animal *topic* or animal
*words*? Two adversarial sets:

- `animal_no_noun` — animal topic, no animal word ("what has a long neck for reaching leaves")
- `animal_word_not_animal` — animal word, not animal ("the Python programming language")

**There is a clean lexical → semantic transition at ~layer 19:**

```
            no_noun↑   word_not_reject↑   verdict
L 8-L18       1.00          0.13-0.40     LEXICAL  (flags "Python" as animal)
L19                                       transition
L24-L31       0.73-0.87     0.80-0.93     SEMANTIC
```

**This overturned the naive layer ranking.** A clean-set sweep said L10 was best
(held-out accuracy 1.00, L28 only 0.93). On adversarial data L10 scores `word_not_reject =
0.27` — building there would have shipped a gate that **emoji-fies Python tutorials**.

> **`d′` and clean-set accuracy actively point at the shortcut.** They are the wrong metric.

### v2 (L26, `d̂` trained adversarially)

Positives = animal + no-noun; negatives = other + decoys.

```
                       v1 (L28)   v2 (L26)
d′                        4.35  →   3.68     ← DROPPED
decoy leakage             6.8%  →   0.3%     ← while performance IMPROVED
```

Direct evidence `d′` was measuring the confound: v1's higher separability was partly the easy
lexical shortcut.

| set | want | v2 |
|---|---|---|
| animal_test | emoji | **91.4%** |
| other_test | prose | **0.0%** |
| animal_no_noun_2 (held-out) | emoji | **90.1%** |
| animal_word_not_animal_2 (held-out) | prose | **0.3%** |

### Positive control — the result that makes the above mean something

"0% emoji on *Jaguar E-Type*" is only evidence of gating if the prompt *can* be emoji-fied.
So: same unit, same α, **gate forced permanently ON**, same decoys → **85.8% emoji**:

```
Firefox web browser        → 🌐🦊🔥🔥🔥
Panda Express restaurant   → 🐼🍜🍗🍜🍜
Penguin Books              → 📚🐧📚🐧📚
Octopus Energy             → 🐙🔋🏠📱🌍
```

The model *has* 🦊🐼🐧🐙 and will emit them for these exact prompts. The conditional 0.3% is
**genuine gating**, not missing vocabulary.

---

## 4. Degeneracy, the triangle, and the limit of fixed writes

### The ceiling control

"Is `🦒🦒🦒` the circuit failing, or is there no good emoji answer to *giraffe evolution*?"
Generate the same prompts under the **true system prompt, unedited**:

```
giraffe evolution   ceiling: 🦒🌿🌍🦕🦖🧬🕰️🌱🌳🦋🐝🌞🌙   (13 unique)
                    circuit: 🦒 **🦒🦒🦒🦒🦒🦒🦒🦒🦒      ( 1 unique)
```

**Mean ceiling diversity 0.890, unique ≈ 11; only 3/40 prompts have a genuinely low ceiling.**
So variety is achievable ~92% of the time. The 3 unfair ones (kangaroo reproduction, bees→honey,
train dogs) are excluded from later evaluation — there the reference itself repeats.

### Measuring degeneracy correctly

A token-id repeat test is **broken**: every pictograph shares lead byte 9008, so perfectly
varied output scores ~0.5 "degenerate". `corr(my_frac, rep4) = 0.22`. Fixing to **glyph-level**
labels (repeat within last k=4 *glyphs*) — unit-tested to 0.00 on varied output and 0.50–0.80
on real loops:

```
b-vs-c d′:   1.38 (token-id)  →  2.50 (glyph-level)      ≈89% per-token, AUC 96%
```

### The triangle is real

Three clouds, on-policy, per generated token at L26: **a** = prose, **b** = good emoji,
**c** = degenerate.

```
cos(v_ab, v_cb)    = +0.071     ← near-orthogonal: a genuine second axis
cos(v_cb, d̂_gate)  = +0.083     ← and orthogonal to the animal condition
‖v_cb‖/‖v_ab‖      =  0.502
```

(3 clouds → only **2** independent difference vectors; `v_ac = v_ab − v_cb`.)

### Damper v1 — failed, and why

Wrote `−v̂_emoji`. Both units then used the *same vector*, so despite the triangle the two
**actions** spanned a 1-D line. Backing out along `−v̂_emoji` retreats toward cloud (a) = prose:

```
a2    glyphs  unique   emoji%
0.0     12.7     1.7    90.6%
0.3      4.3     1.8    73.3%     ← unique FLAT; the "diversity gain" is pure truncation
```

### Damper v2 — write the third side, `+v̂_cb`

Both units live in **L26's MLP** — drive = unit **1467**, damper = unit **938** — reading the
same `post_attention_layernorm(h)` and writing the same residual. They do not interfere
(`down_proj @ act` sums over units; the gates are near-orthogonal, `cos(v_cb, d̂_gate)=0.083`).
**2 units × 3 vectors × 2560 = 15,360 params** (the 7,680 in §3 is the drive alone).

```
mode     a2   uniq  nglyph  emoji%   leak%
OFF     0.0    1.5    12.6   88.7%    0.0%
v_cb    0.3    2.6    11.4   55.0%    0.0%     ← real: unique rises, length held
v_cb    0.6    2.9    10.3   36.4%    0.0%
v_cb/g  0.3    1.7    12.0   81.7%    0.0%     ← RMSNorm-corrected is WORSE (unexplained)
```

Genuine but small: **unique 1.5 → 2.6 against a ceiling of 11 (~11% of headroom)**, costing
emoji purity (89→55%), and the added glyphs are **semantically wrong**.

**The decisive tell: `🐻` appears as the second glyph for kangaroos, bees *and* wolves**
(also 🦓 for giraffes, 🐠 for octopus, 🐯 for ants).

> `v_cb` is a difference of cloud **means**, so pushing along it injects the *centroid* of the
> good-emoji cloud — identically for every prompt. **Variety is not a state, it is a relation
> to what was just emitted.** No fixed vector can track a referent that moves each token.

---

## 5. LoReFT comparison — the controlled variable is input-dependence

LoReFT's `Φ(h) = h + Rᵀ(Wh + b − Rh) = (I − RᵀR)h + Rᵀ(Wh + b)` is **affine in h**: it reads
the full `h` through `W` and writes into the rank-4 `R` subspace, so the push can be
*proportional to* animal-ness. A conditional is expressible — but **linearly, with no
threshold** (unlike SiLU).

Dataset: **25 emoji targets** (ceiling answers) + **45 prose targets** (non-animal, *including*
the decoys — "Dove chocolate", "Python language"). Trained L17, rank 4, `positions=all`,
best epoch 5 (held 0.570). No system prompt (per the earlier A/B finding).

```
                      animal_test  other_test  no_noun_2  word_not_2   unique
                       (want 🙂)   (want prose) (want 🙂) (want prose) (ceil 11)
system prompt              98%          —           —          —        ~11
hand-built circuit       91.4%        0.0%       90.1%       0.3%     1.5–2.6
conditional LoReFT       66.7%        8.3%       74.1%       8.4%     5.4–6.1
```

**Exactly complementary, as the expressivity analysis predicted (logged before the run):**

| | gate (IF) | write (THEN) | result |
|---|---|---|---|
| hand-built circuit | **nonlinear** (SiLU threshold) | **fixed** column | crisp conditionality, no variety |
| LoReFT | **linear/affine**, graded | **state-dependent** `Rᵀ(Wh+b)` | real variety, leaky gate |

LoReFT's emoji are *apt*, not centroid noise:
`chameleon → 🦎🌿🎨👁️🌈🕷️🌳🐍🦂🐝🐞🦗` (🎨 colour-change, 👁️ vision).
But it under-fires (giraffe evolution → prose) and leaks (`Panda Express → 🐼🍽️🥢`).
**It is not immune to loops either**: `ravens → 🦅🧠🦅🧠🦅🧠`.

Surprise: LoReFT scored **higher** on `no_noun_2` (74.1%) than `animal_test` (66.7%) — its
learned gate generalised to animal-topic-without-animal-word unaided. Both methods
independently found the semantic feature rather than the lexical one.

---

## 6. Findings

1. **The cute demo needs no weight *reasoning*.** Tokenizer bookkeeping (which row opens the
   emoji block) plus one geometric measurement (the residual cone) suffices, at every depth
   L8–L28. A flashy "LLM edited weights and behaviour changed" result is *not* evidence of
   weight-space reasoning — the critique's point, demonstrated from the **write** side.
2. **The boundary is conditionality, not editing.** Bookkeeping suffices when the target is a
   *coordinate of the output parameterisation* (token 9008) and the behaviour is
   unconditional. "Animal topic" has no coordinate — it must be *constructed* from data, and
   that is where interpretability does load-bearing work.
3. **Localisation informs editing — but only when the probe is adversarially validated.**
   `d′` and clean-set accuracy pointed at a lexical shortcut; the adversarially-trained
   direction had *lower* `d′` and 20× less leakage. This qualifies Hase et al. (2301.04213,
   "localisation does not inform editing"): with a validated probe here, it did.
4. **Fixed-direction writes cannot produce variety**, because a mean-difference direction
   injects a centroid identically for every input. Confirmed by the universal `🐻`.
5. **Gate expressivity and write expressivity are separable and orthogonal.** The ideal object
   is a **hard gate driving a state-dependent actuator** — neither of the two things built here.

---

## 7. Caveats / open threads

- Clouds (a,b,c) and `v_ab` were measured **under the edit**, so they partly reflect the
  intervention, not the base model. The b-vs-c contrast is clean (same edit both sides).
- **`diversity = unique/total` rewards truncation.** Always report unique-count and length.
  Several earlier calls in this thread (e.g. "L8 is most diverse") were artifacts of this.
- Eval sets are 12–20 prompts → ±5–7 pp.
- **`cos(v_ab, v̂_emoji)` never measured.** The drive used `v̂_emoji` (available before any
  generation) rather than the measured `v_ab`; it may be a suboptimal axis.
- `v_cb` is used as *both* gate and write for the damper — the assumption that the direction
  which detects degeneracy is the right one to fix it was never tested separately.
- **The damper's layer was never swept — probably the biggest cheap win left.** Both units sit
  in L26's MLP (drive = unit 1467, damper = unit 938), reading the same
  `post_attention_layernorm(h)` and writing the same residual. That was convenience, not a
  considered choice. Two reasons L26 is likely wrong for the damper:
  (a) *readability* — animal-ness is linearly available from L7 on (swept: ≥0.87 across 25
  layers), but degeneracy **accumulates during generation** and its readability-by-layer was
  never measured at all; (b) *latency* — by L26 the token is nearly decided, leaving ~6 blocks
  for a correction to take effect. The drive wants to be late (early writes destroy semantics,
  §2b), but a controller acting on an error signal plausibly wants to be **earlier**, so the
  model has layers left to route around the loop.
  Nothing forces them to share a layer: the residual stream carries both signals forward, so
  the damper could read at L20 and write at L22 with the drive left at L26. Given how much the
  drive's depth mattered (12× α ladder; semantic grounding collapsing by L8), this is the most
  likely improvement remaining in the fixed-vector design. The read-side half is cheap — the
  same all-layers hook used in `cond_read_sweep.py`, applied to the b-vs-c contrast instead of
  animal-vs-other.
- LoReFT's 25/45 class imbalance (worse by token count, prose targets are longer) was not
  controlled; its under-firing may be a training artifact, not a mechanism limit.
- `v_cb/g` (RMSNorm-corrected) underperformed raw `v_cb` — unexplained.
- Not tested: multi-turn, animal topic arising mid-conversation, or any capability benchmark.

---

## 8. Artifacts

**Scripts** (`scripts/wsp/`): `probe.py` · `emoji_edit.py` · `emoji_mlp_edit.py` ·
`emoji_mlp_depth.py` · `emoji_conditional.py` · `cond_read_sweep.py` · `lexical_control.py` ·
`cond_v2.py` · `degen_precheck.py` · `degen_precheck2.py` · `degen_samples.py` ·
`cond_damper.py` · `cond_damper2.py` · `three_way_table.py` · `build_cond_dataset.py` ·
`eval_cond_loreft.py`

**Weight patches** (`data/wsp/`, originals never clobbered):

| file | size | what |
|---|---|---|
| `emoji_wsp_patch.pt` | 22 KB | `lm_head[9008]` row, β=0.55 |
| `emoji_mlp_L28_patch.pt` | 649 KB | rank-1 `down_proj` write, α=0.45 |
| `emoji_conditional_patch.pt` | 59 KB | the 3-vector conditional circuit |
| `data/reft/mine/emoji_cond_L17.pt` | 63 KB | conditional LoReFT (rank 4, L17) |

**Data**: `cond_prompts.json` (30/15 animal, 30/15 other) · `lexical_control.json` +
`lexical_control_heldout.json` (adversarial) · `degen_precheck2.json` (ceiling) ·
`data/reft/mine/emoji_conditional.json` (70 train examples)
