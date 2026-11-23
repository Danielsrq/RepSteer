# Installing a conditional sentence into an LLM's weights, in closed form

A self-contained explanation of the method used in `scripts/wsp2/`, written as the derivation
arc that actually clarified it, including the places where the first explanation was wrong or
misleading. Model: Qwen3.5-4B (32 layers, d_model 2560, d_ff 9216, tied embeddings).

**Goal.** "Emit a fixed sentence when the topic is X, behave normally otherwise", installed by
editing weights -- no fine-tuning, no system prompt, no runtime hook.

---

## 1. The object being edited

Qwen's MLP is SwiGLU, three matrices:

```
x  in R^2560   (residual stream at layer L)
       |-- W_gate --> SiLU --|
       |                     (*)      <- ELEMENTWISE product
       |-- W_up -------------|
                             |
                        k in R^9216        <- "the key"
                             |
                          W_down          [2560 x 9216]
                             v
                        o in R^2560       <- added back into the residual
```

So `k = SiLU(W_gate x) (*) (W_up x)` and `o = W_down k`.

Note `k` is a **nonlinear** function of `x` (SiLU, and a product). This matters: it is why a
single key can encode a *conjunction* like "food topic AND at a clause break". A purely
linear projection could not.

Following Geva et al., read `W_down` as a **linear associative memory** `k -> W k`: keys in
R^9216, values in R^2560. We INSERT associations into it. Only `W_down` at one layer is
touched; `W_new = W + Delta`. Nothing is *replaced*; Delta is an additive perturbation.

### Keys are measured, and one prompt yields many

Hook `W_down` and run a forward pass: its input is a `[T x 9216]` matrix -- **one key per
token position**. A 41-token sequence gives 41 keys from one forward pass. This is why
"416 trigger keys" never required 416 hand-written prompts: it is 32 prompts x 13 positions.

---

## 2. Two stages. They are never solved together.

This is the single most important structural fact, and the thing that is easiest to miss.

```
+-- STAGE 1 ------------------------------------------------------------+
|  Delta DOES NOT EXIST YET. The model is unmodified.                   |
|  A forward hook SUBSTITUTES the MLP's output with a free vector V.    |
|  Optimise V by gradient descent. 25 Adam steps.                       |
|  unknowns: 13 x 2560 = 33,280                                         |
|  OUTPUT: v*_0 .. v*_12, now FIXED CONSTANTS                           |
+-----------------------------------------------------------------------+
                                  |
+-- STAGE 2 ------------------------------------------------------------+
|  v* is now a KNOWN NUMBER, so r_i = v*_j - W k_i is fully computable. |
|  Solve a linear least-squares for Delta. ONE closed-form solve.       |
|  unknowns: 23.6M                                                      |
|  OUTPUT: Delta                                                        |
+-----------------------------------------------------------------------+
```

| stage | question | method |
|---|---|---|
| 1 | *what should the MLP output be here?* | gradient descent (nonlinear, needs the network) |
| 2 | *what weight change makes the MLP output that?* | linear least squares (exact, closed form) |

**The split is what buys the closed form.** Given `v*`, stage 2 is *linear* in Delta, so it
has an exact solution and the "leave these 1087 keys alone" constraint can be weighted
explicitly. Optimising weights and behaviour jointly would just be fine-tuning: no closed
form, no explicit preservation term.

`v*` is an **activation**, not a weight. It lives nowhere in the model. It is scaffolding,
discarded after stage 2. The only artifact that ships is `W + Delta`.

---

## 3. Stage 1: `v*` by optimisation

`v*` is **not measurable.** There is no forward pass anywhere that hands it to you. If there
were, you would already have a model that says the payload -- which is the thing you do not
have.

```python
V = v_init.clone().requires_grad_(True)      # 13 x 2560 = 33,280 FREE PARAMETERS
opt = torch.optim.Adam([V], lr=0.5)         # V is the ONLY thing optimised
                                            # model.requires_grad_(False): W frozen
for step in range(25):
    for q in prompts:                        # 6 trigger prompts, loss averaged
        seq = prompt_tokens(q) + payload_tokens          # teacher-forced
        hook.force = (P-1, V)                # OVERWRITE W_down's output at
        out = model(seq)                     #   positions P-1 .. P-1+12 with V
        loss = cross_entropy(out.logits[P-1 : P-1+13], payload_tokens)
        loss.backward()                      # gradient flows back to V
    opt.step()
    clamp(||V[j] - v_init[j]|| <= clamp * ||v_init[j]||)
```

The gradient path:

```
V -> written into the residual at L26 -> layers 27..31 -> final norm -> lm_head
  -> logits -> cross-entropy against the payload
     |
     +-- dL/dV flows back through those 6 layers
```

The model tells us, through its own gradients, what residual-stream content at layer 26 would
lead to emitting this sentence. We are asking the network to invert itself by descent.

`v_init = W k_bar` -- the MLP's actual mean output. At step 0, `r = 0`, Delta would be zero,
the model behaves normally, and the payload has probability ~0.19% per token. Descent walks V
away until the payload is at ~99.99%. So:

> **`r` is literally the displacement the optimiser travelled, measured from "change nothing".**

Averaging over 6 different prompts is what makes the 13 vectors prompt-independent.

### The off-by-one

Forcing position `p` changes the logits at `p`, which predict the token at `p+1`. So with the
payload at sequence indices 10..20, the KEYS are at 9..19:

```
idx:     ...  8   9  | 10    11    12  ...  20 | 21 ...
token:        ?   ?  |"sur" "prise"" you" "in!"|  ?
                  ^
            assoc 0 key -- the LAST PROMPT TOKEN
```

Association 0's key is the assistant-header token. That is why **association 0 is the only
conditional one**: it is the only key that sees the prompt and not the payload. Associations
1..12 are gated by "I am mid-payload", which is self-sustaining and self-terminating.

---

## 4. Stage 2: Delta by closed-form least squares

Notation: `d = 2560`, `D = 9216`, `W in R^{d x D}`.
`K_t in R^{D x n_t}` trigger keys as columns (n_t = 416);
`K_p in R^{D x n_p}` preserve keys (n_p = 1087).

Targets:
- trigger key i:  `(W+Delta) k_i = v*_{a(i)}`, i.e. `Delta k_i = r_i = v*_{a(i)} - W k_i`
- preserve key j: `Delta k_j = 0`      (target IS the current output: leave it alone)

Objective:

```
min_Delta   sum_{i in t} ||Delta k_i - r_i||^2  +  lam sum_{j in p} ||Delta k_j||^2  +  eps ||Delta||_F^2
            \-------- fire ---------/              \------ do no harm ------/          \- minimum norm -/
```

### The one idea: each ROW is a ridge regression

Let `delta_m in R^D` be row m of Delta. Read the vector equation one component at a time:

```
Delta k_i = r_i        (a vector equation in R^2560)
   =>  delta_m . k_i = r_i[m]        <- scalar = scalar.  The [m] is doing the work.
```

`r_i` is a vector; `r_i[m]` is its m-th COMPONENT, a scalar. `delta_m . k_i` is a dot product,
a scalar. Row m is fit against component m of every `r_i` -- 1451 scalars, one per key.
Across all 2560 rows, every component of every `r_i` is used exactly once.

```
vector form:  Delta k_i = r_i                 n = 1503 vector equations
scalar form:  delta_m . k_i = r_i[m]          1503 x 2560 = 3.8M scalar equations
row m sees:                                   1503 of them
```

So **solving for a [2560 x 9216] matrix = 2560 independent ridge regressions that share a
design matrix.** That is the whole trick. Everything below is bookkeeping.

### The algebra (steps 2-4 add no ideas, only efficiency)

Stack `K = [K_t, sqrt(lam) K_p] in R^{D x n}`, and let `Y in R^{n x d}` hold the targets
(column m = the `y_m` for row m). Then

```
1.  per row:          delta_m = (K K^T + eps I_D)^-1 K y_m
2.  stack 2560 rows:  Delta   = Y^T K^T (K K^T + eps I)^-1
3.  Y^T = [R | 0]  (zero on preserve keys), so Y^T K^T collapses:
                      Delta   = R K_t^T (K K^T + eps I)^-1
4.  kernel identity   K_t^T (K K^T + eps I)^-1 = S (K^T K + eps I)^-1 K^T,   S = [I 0]
                      Delta   = R S (K^T K + eps I)^-1 K^T
```

Step 4 is purely computational: `K K^T` is 9216 x 9216 (340 MB); `K^T K` is **1503 x 1503**.
Same answer, one `torch.linalg.solve`.

### Why rank(Delta) <= 416

`Delta = R @ (anything)` and `R in R^{2560 x 416}`, so every column of Delta is a combination
of R's 416 columns. Expanding:

```
Delta = sum_i  r_i (x) c_i          Delta k = sum_i r_i (c_i . k)
                |      |
             WRITE    READ
            R^2560   R^9216
         residual    key space
```

So there ARE contrast vectors -- **416 of them** -- and they have the same shape as a
hand-built read/gate/write circuit. The difference is how they arise: `c_i` is a weighted
combination of **all 1503 keys**, weights chosen by the inverse Gram so every constraint holds
at once. Trigger keys get positive weight, preserve keys negative. **Difference-of-means (DiM)
is the special case where you fix those weights by hand.**

### lam and eps

- `lam` scales the preserve keys by `sqrt(lam)`, turning `Delta k_p = 0` from hard to weighted.
  It is the exchange rate between firing and not breaking anything. It is NOT a strength knob:
  `||Delta||` *decreases* as lam rises while the fire rate drops.
- `eps` picks the minimum-norm solution among infinitely many.

### Underdetermination is a FEATURE

Both stages are underdetermined, and both are regularised:

| stage | unknowns | constraints | what makes it well-posed |
|---|---|---|---|
| 1 | 33,280 | 78 target tokens (13 x 6 prompts) | init at `W k_bar` + norm ball (`clamp`) |
| 2 | 23.6M | 3.8M scalar equations | `eps ||Delta||^2` -> minimum norm |

1503 keys span at most a 1503-dim slice of the 9216-dim key space; `eps` sets Delta to zero on
the other ~7700 dimensions. **That zero is what stops the edit touching unrelated inputs.**
Fully determining all 23.6M entries would need 9216 independent keys per row and Delta would be
dense in key space -- far more destructive. More data does not "fit better"; it does two
distinct things:

| add more... | effect | observed |
|---|---|---|
| trigger keys (more prompts) | the **condition generalises** | 6 prompts -> 33% fire; 24 -> 69%; 32 -> 90-95% |
| preserve keys | **less collateral** | the in-answer fix |

and both raise `rank(Delta)` and `||Delta||`, making the edit louder and easier to detect.

---

## 5. Why not MEASURE `v*` instead of optimising it?

A natural proposal: run a forward pass and read the "true counterfactual" output off the model
rather than optimising. The steelmanned version is a real technique -- it is what attempt 1 did
with `v_ab`:

1. Scaffold the behaviour so the model really does emit the payload (e.g. a system prompt
   "Reply with exactly: ...", or an existing edit).
2. Read the MLP output at those positions. It is now, by construction, a vector present when
   the payload was produced.
3. Transplant it as `v*`.

**Why it is weaker here:**

- **It measures presence, not sufficiency.** The measured `W k` is only *part* of what produced
  the payload; the rest came from the context (the system prompt sitting in the KV cache).
  Remove the context and the measured vector alone may not suffice. We observed exactly this in
  attempt 1: `v_ab` worked but needed 2x the strength (c=4 vs c=2) and gave worse output.
- **Optimisation targets sufficiency directly.** `v*` is found by requiring that substituting
  it *alone*, with no system prompt, produces the payload. That is strictly stronger than
  "this was present when the behaviour happened."
- **It is contaminated by the scaffold.** The measured activation encodes "I am obeying an
  instruction to recite", which is not the state we want at deployment.

**Why reading the output at `i-1` specifically does not work:** the MLP output at `i-1` on the
unmodified model is `W k_{i-1}` -- which is precisely what the model *already does*, and what it
does is continue with normal prose, not the payload. You would get `r = 0`. There is nothing to
read off.

### The on/off-policy question, properly

The worry: teacher forcing is off-policy, so keys collected at payload positions come from a
token stream the model would never generate.

**For the keys, teacher forcing is not an approximation -- it is correct.** At deployment, IF
the edit works, the model WILL have generated exactly those tokens, so `prompt + payload[:j]`
is exactly the context that will exist. Teacher forcing *constructs the deployed trajectory*.
It is "off-policy" only with respect to the *unedited* model, which is irrelevant -- we are
building the edited one. This is self-fulfilling, and verified empirically:
`assoc-0 key consistency (chain vs prefill): rel diff 1.26e-02`.

The recursion worry is also correct, and it is exactly why the measure-it approach does not
close: to get a genuinely on-policy output at position i you need the model to have generated
`payload[:i]` by itself, which it will not do until the edit exists. **You would need the edit
in order to collect the data to build the edit.** Teacher forcing breaks the circularity by
*asserting* the trajectory instead of sampling it.

**But there IS a residual on-policy problem, just not where expected.** The edited model's
actual trajectory can differ from the forced one in small ways -- notably tokenisation at
boundaries: the same payload text tokenises differently after `.` than after `**`. When that
happens the chain keys do not match and the chain breaks or misfires. We hit this concretely:
it broke the loop-detection metric, and it is a real failure mode. The standard remedy is to
**iterate** -- collect keys from the *edited* model and re-solve (MEMIT does this). We have not
done that; it is the clearest open improvement.

---

## 5b. Provenance: what is borrowed, what is not (papers read and verified)

Verified by reading ROME (arXiv 2202.05262, 33pp), MEMIT (2210.07229, 21pp) including
appendices, and BadEdit (2403.13355) via summary.

### Borrowed

| component | source |
|---|---|
| `W_down` as a key-value associative memory | Geva et al., "Transformer FF Layers Are Key-Value Memories" |
| two-stage: optimise an activation `v*`, then closed-form weight update | **ROME** (sec 3.1). Adam lr 0.5, <=20 steps, early stop at 5e-2 |
| many associations solved jointly as least-squares with a preservation term | **MEMIT** Eqn 14: `Delta = R K_1^T (C_0 + K_1 K_1^T)^-1` |
| "backdoor injection as lightweight model editing", tiny data | **BadEdit** (ICLR 2024), 15 samples, ROME/MEMIT-based. Its "duplex editing" `Delta = Delta_backdoor + Delta_clean` is the same trigger+preserve split |

### NOT in any of them (verified ABSENT)

1. **Semantic-class trigger.** Both papers key every edit on a *named subject*
   (`k(x_j + s_i)`). BadEdit uses rare lexical tokens (`"tq"`), explicitly *chosen to
   minimise accidental activation*. Conditioning on a concept, with adversarial lexical
   decoys as explicit constraints, is absent from this line.
2. **A chain across consecutive OUTPUT positions.** MEMIT: *"our focus will be on states
   representing the last subject token S."* Single position in both. The `assoc=first`
   ablation here quantifies why a chain is needed (one association carries ~2 tokens).
3. **Self-generated (infix) trigger position** — conditioning on the model's own
   mid-generation state. Absent everywhere.
4. **On-policy rollout iteration.** ROME is a single closed-form shot. MEMIT recomputes keys
   **across layers** (Algorithm 1 lines 6-8) from *prompts*, never from rollouts. BadEdit's
   incremental batching is over fixed data.

NOTE: an earlier version of these notes claimed "MEMIT does this" of rollout iteration. That
was wrong and is retracted. Also, neither paper states a completion length in tokens (ROME R5,
MEMIT M4 both ABSENT); the inference that they insert short entity names is reasonable from
CounterFact but is not a cited fact.

### FOUR places this implementation is WORSE than the papers

These are the most likely causes of the collateral damage fought throughout section 3, and all
four are orthodox fixes that were skipped:

| | papers | here |
|---|---|---|
| preservation | `C = E[k k^T]` from **100,000 Wikipedia/Wikitext samples**. ROME uses **no lambda** (hard equality constraint); MEMIT uses one scalar `lambda ~ 1.5e4` | **1035 hand-picked keys**. A different object, not just a small sample |
| `v*` regulariser | **KL "essence drift" term**, ROME Eqn 4b, lambda=100: penalises divergence of predictions on a neutral prompt | **nothing** -- only a norm ball (which ROME does NOT use; R7) |
| layer | MEMIT **spreads** across `R={3,4,5,6,7,8}` for GPT-J, residual divided evenly (Eqn 20) | **one** layer. This is very likely why `||Delta||/||W|| = 20-57%` |
| key robustness | key averaged over **20 random prefixes** (ROME Eqn 3: ten of length 5, ten of length 10) | single key per prompt |

Also worth noting the layer choice differs sharply in character: ROME picks layer 18 of 48
(~37%) by causal tracing, as *"approximately when MLP outputs begin to switch from acting as
keys to values"*; MEMIT uses layers 3-8 of 28 (~11-29%). This work uses L26 of 32 (~81%),
found by sweeping behaviour -- and the read-side diagnostic (`d'`) ranked layers *backwards*.

## 6. Common confusions, collected

| confusion | correction |
|---|---|
| "we replace `W_down` with Delta" | we **add**: `W_new = W + Delta` |
| "the 9216-vector is `W_up`'s output" | it is `SiLU(W_gate x) (*) (W_up x)` -- the nonlinearity matters |
| "rank 416 means 416 rows were edited" | no rows are selected; Delta is **dense**, all 23.6M entries change. Rank is about its *image* |
| "the contrast vectors live in residual space" | `c_i` lives in **key space** R^9216; `r_i` lives in residual space R^2560 |
| "`v*` is the MLP's output, so `v* - W k = 0`" | `v*` is the output we **want**; `W k` is what it **does**. They differ, which is the whole point |
| "`delta_m . k_i = r_i` is scalar = vector" | it is `r_i[m]`, the m-th component. Scalar = scalar |
| "`v*` and Delta are solved jointly" | two sequential stages; `v*` is a fixed constant before Delta is touched |
| "everything that is not a trigger key is a preserve key" | **this is the CORRECT design.** The implementation hand-picked preserve positions and omitted everything inside the model's own generated answer -- which is why the payload re-fires mid-answer and why leakage appeared at 160 tokens but not 56 |

---

## 7. What is authored vs derived

| | count | origin |
|---|---|---|
| prompts | 32 trigger + 70 general + 25 decoy | hand-written -- **this is where the concept is defined** |
| payload string | 1 | chosen |
| `v*` vectors | 13 | Adam, 25 steps |
| keys `k_i` | **1503** | **measured** from ~127 forward passes |
| corrections `r_i` | 416 | **derived**: `v* - W k_i` |
| Delta | 23.6M entries | one closed-form solve |

Only three things are authored. Everything with a large count is measured or derived.

---

## 8. Scale check from the real run

```
||W_down||_F           = 45.0        (per-entry std 0.0093)
||Delta||_F            = 8.9 - 16.5  ->  ||Delta||/||W|| = 20-37%
||v_init|| = ||W k||   = 4.2 - 7.5   per association
||r|| at clamp 1.5     ~ 7.4
```

The required correction is **~1.5x LARGER than the MLP's own output** at that position. `v*` is
not a nudge. And the clamp sweep shows this is not waste:

```
clamp   ceiling   fire   ||Delta||   KL       fit
0.25       0%      0%      6.1      0.0010   0.255
0.50      80%     25%      6.8      0.0013   0.180
1.50     100%     90%     12.3      0.0042   0.100
```

At clamp 0.25 the *ceiling* is already 0% -- the payload cannot be driven even by forcing `v*`
directly, so the failure is at the activation level, not in the solve. And the ceiling-to-fire
gap widens as the edit gets quieter (100->90 at clamp 1.5; 80->25 at clamp 0.5): when `v*` only
barely suffices, the solve's fit error is enough to lose the behaviour. **The loudness is margin
against the approximation error of the weight solve.**
