# Convergence & Stopping in Multi-Agent LLM Systems and Reasoning Chains

**Survey date:** 2026-09-07
**Question:** How do systems DETECT that a multi-agent debate / reflection loop / single
chain-of-thought has "converged" (positions settled, further generation adds nothing) and
decide to STOP? Organized into two technique families:
- **(A)** prompt / orchestration / output-level signals (fixed rounds, judge LLM, voting,
  answer-stability over produced text).
- **(B)** activation / representation-engineering / internal-state signals (probing hidden
  states, logit-entropy / confidence stopping, reading the internal "committed answer").

> **Citation-count caveat:** Semantic Scholar Graph API returned HTTP 429 (rate-limited)
> for the entire research session; **no verified citation counts were obtainable.** All
> counts are `N/A (S2 rate-limited, 2026-09-07)`. Within-family ordering is by
> prominence/recency, NOT verified counts. arXiv IDs were reported as web-verified by the
> research agent; the 2025–2026 IDs are worth re-checking before citing. To fetch counts:
> `https://api.semanticscholar.org/graph/v1/paper/search?query=<title>&fields=citationCount`
> (use an API key; unauthenticated is what got throttled).

---

## HEADLINE FINDING (the gap)

- **Single-chain** internal-state stopping (family B) is real and fast-growing in 2025–26.
- **Multi-agent** convergence detection is **overwhelmingly output-level (family A)** — fixed
  rounds, judge LLMs, answer voting / statistical answer-stability.
- **No paper found probes hidden states / residual stream / logits to decide a *debate* has
  reached consensus and should stop.** The only cross-agent internal-representation work
  (collusion detection, arXiv:2604.01151) targets deception, not convergence, and itself
  calls cross-agent internal representations "unexplored."
- ⇒ **Representation-level consensus/convergence detection for multi-agent systems is an
  open, essentially untouched niche.** The single-chain toolkit (committed-answer probes,
  entropy/confidence early-exit) is ready to port.

---

## FAMILY (A) — prompt / orchestration / output-level

| # | Paper (first author, year, venue) | arXiv | Mechanism | Citations |
|---|---|---|---|---|
| A1 | Self-Consistency (Wang, 2022, ICLR'23) | 2203.11171 | majority-vote over sampled CoT chains; fixed budget (agreement-across-samples primitive) | N/A |
| A2 | Multiagent Debate (Du, 2023, ICML'24) | 2305.14325 | multiple LLMs debate **fixed # rounds**; read final answer | N/A |
| A3 | MAD / Divergent Debate (Liang, 2023, EMNLP'24) | 2305.19118 | two debaters + **judge/moderator LLM** decides when to extract answer | N/A |
| A4 | Reflexion (Shinn, 2023, NeurIPS'23) | 2303.11366 | reflect-and-retry; stop on **task-success** signal / max trials | N/A |
| A5 | Self-Refine (Madaan, 2023, NeurIPS'23) | 2303.17651 | loop until model's **own feedback says "good enough"** / cap | N/A |
| A6 | LLMs Cannot Self-Correct Reasoning Yet (Huang, 2023, ICLR'24) | 2310.01798 | *critique*: ad-hoc self-stops unreliable, can **degrade** answers | N/A |
| A7 | ReConcile (Chen/Saha/Bansal, 2023, ACL'24) | 2309.13007 | multi-round round-table; **stop on consensus** + confidence-weighted (verbalized) vote | N/A |
| A8 | Adaptive-Consistency (Aggarwal, 2023, EMNLP'23) | 2305.11860 | sequential sampling; **halt when dominant-answer posterior > threshold** (~7.9× fewer samples) | N/A |
| A9 | Voting or Consensus? (Kaesberg, 2025, Findings ACL'25) | 2502.19130 | compares 7 termination protocols; voting↑reasoning, consensus↑knowledge | N/A |
| A10 | Aegean / reaching agreement (Ruan, 2025, NUS) | 2512.20184 | classical distributed **consensus quorum detection**; guards vs *transient agreement*; 1.2–20× faster | N/A |
| A11 | MAD Judges + Adaptive Stability Detection (Hu, 2025) | 2510.12697 | **Beta-Binomial mixture + KS test** on round-to-round verdict distribution | N/A |
| A12 | Answer Convergence early-stop (2025, EMNLP'25) | 2506.02536 | halt single CoT when **intermediate answers stop changing**; + light stop classifier | N/A |
| A13 | Free-MAD: Consensus-Free Multi-Agent Debate (2025) | 2509.11035 | counterpoint: pick answer **without** requiring agreement | N/A |
| A14 | Multi-Agent Debate Strategies: Survey/Taxonomy (2026) | 2607.26212 | taxonomy incl. termination/decision protocols | N/A |
| A15 | AI Safety via Debate (Irving, 2018, OpenAI) | 1805.00899 | foundational; human/judge verdict on fixed transcript | N/A |

**(A) trend:** fixed-rounds/judge → **statistical answer-stability / quorum tests**
(Adaptive-Consistency, Adaptive Stability Detection, Aegean), explicitly guarding against
*premature/spurious* consensus ("agents agreed too fast").

---

## FAMILY (B) — activation / representation / internal-state (all SINGLE-chain)

| # | Paper (first author, year) | arXiv | Internal signal | Citations |
|---|---|---|---|---|
| B1 | Reasoning Models Know When They're Right (Zhang, 2025) | 2504.05419 | **probe on hidden states** reads committed/likely-correct answer *before* written → exit (−24% tokens) | N/A |
| B2 | Certaindex / Dynasor "Probe-In-The-Middle" (Fu, 2024) | 2412.20993 | force-decode internal committed answer mid-CoT + semantic entropy; certainty → cut budget (~50%) | N/A |
| B3 | DeepConf (Meta, 2025) | 2508.15260 | **token logprob confidence** (group/tail) filters & early-stops traces online; no training | N/A |
| B4 | Think Just Enough (2025) | 2510.08146 | **Shannon entropy of token logprobs** as stop signal (~25–50% compute saved) | N/A |
| B5 | Entropy After `</think>` (EAT, 2025) | 2509.26522 | next-token **entropy decreases & stabilizes** at plateau → stop | N/A |
| B6 | CCPS (2025, EMNLP'25) | 2505.21772 | perturb final hidden states, probe **representation stability** for correctness | N/A |
| B7 | Early Stopping via Confidence Dynamics (2026) | 2604.04930 | track **confidence dynamics** over trace to stop | N/A |
| B8 | Hidden States as Early Signals (2026) | 2601.09093 | probe **step-level hidden states** to evaluate/prune traces | N/A |
| B9 | Temporal Predictors of Outcome (2025) | 2511.14773 | probe **internal states over time** to predict eventual outcome | N/A |

Related family-(B) internal-state confidence/hallucination probes (convergence-adjacent, not
fully vetted): "Tracing the Traces" 2510.10494; "ICR Probe" 2507.16488; "Confidently Wrong"
2607.11414.

### Multi-agent × internal-state (the near-empty cell)
- **Detecting Multi-Agent Collusion Through Multi-Agent Interpretability** (2026, arXiv:2604.01151)
  — only work reading internal activations across *multiple interacting agents*; targets
  **collusion/deception**, NOT convergence; states cross-agent internal representations
  "remains unexplored."
- **Towards Ethical Multi-Agent Systems … Mechanistic Interpretability** (2025, 2512.04691)
  — position/agenda piece; no convergence detector.

---

## Synthesis & gaps

1. **Multi-agent convergence detection is ad-hoc and output-level.** Debates stop by fixed
   rounds (Du), a judge LLM (Liang), or consensus + voting (ReConcile); reflection loops stop
   on self-declared "good enough" (Self-Refine, Reflexion) — which Huang 2023 shows is
   unreliable. Best-practice (A) replaces fixed rounds with statistical answer-stability /
   quorum tests that guard against transient agreement (Adaptive-Consistency, Adaptive
   Stability Detection, Aegean).
2. **Family (B) is real but single-chain only.** Hidden-state committed-answer probes
   (2504.05419, 2412.20993, 2601.09093, 2511.14773) and logit-entropy/confidence early-exit
   (2508.15260, 2510.08146, 2509.26522) detect when *one* chain has internally committed.
3. **The open niche:** no work uses internal representations to detect *multi-agent* consensus.
   Porting (B) to multi-agent — e.g. a probe over each agent's hidden "committed answer" to
   declare genuine (vs sycophantic/transient) consensus, or entropy/confidence agreement
   across agents as a stop signal — is essentially untouched. Reinforced by findings that
   text-level consensus is often premature/spurious ("inter-agent sycophancy", Aegean's
   "transient agreement").

## Follow-ups
- Retrieve citation counts via S2 Graph API **with an API key** (unauthenticated was 429'd).
- Re-verify 2025–2026 arXiv IDs before citing.
- Candidate project: family-(B) multi-agent convergence detector (per-agent committed-answer
  probe + cross-agent entropy/stability), reusing this playground's probing machinery.
