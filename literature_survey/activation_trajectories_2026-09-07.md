# Activation Trajectories in LLMs — Literature Survey

**Survey date:** 2026-09-07
*Scope: how a model's internal activations (residual stream / hidden states) evolve — across
depth (layer-by-layer) and across time (token-by-token during generation) — plus the
representation-engineering / activation-methods toolkit for reading, steering, and modeling
that evolution. Foundational works + recent (2023–2026).*

> **Citation-count note:** counts gathered mainly from the Semantic Scholar (S2) Graph API,
> which was heavily HTTP-429 rate-limited. "(S2 API)" = direct API read (reliable);
> "(S2 snippet / mirror)" = from a search snippet or alphaXiv/emergentmind mirror (approximate);
> "N/A" = unobtainable. **Theme 3 counts are clean S2 API reads (2026-09-07); other themes are
> approximate.** All arXiv IDs verified against live abs pages. Nothing fabricated; preliminary/
> withdrawn/unverified items are flagged.

---

## Theme 1 — Layer-wise trajectories (across depth)
Logit/tuned lens; iterative-inference / residual-stream-as-gradual-refinement; early decoding.

- **1.1 Residual Connections Encourage Iterative Inference** — Jastrzębski et al., 2017 (ICLR'18) · [1710.04773](https://arxiv.org/abs/1710.04773) · ~135 (snippet). Root of the layer-wise-refinement view: residual blocks successively refine (not replace) representations.
- **1.2 Eliciting Latent Predictions with the Tuned Lens** — Belrose et al., 2023 · [2303.08112](https://arxiv.org/abs/2303.08112) · **586 (S2 API)**. Learned affine per-layer refinement of the logit lens → smooth prediction trajectory over depth.
- **1.3 DoLa: Decoding by Contrasting Layers** — Chuang et al., 2023 (ICLR'24) · [2309.03883](https://arxiv.org/abs/2309.03883) · **428 (S2 API)**. Contrasts late- vs early-layer logits at decode time to amplify factual knowledge, cut hallucination.
- **1.4 Do Llamas Work in English? (Latent Language of Multilingual Transformers)** — Wendler et al., 2024 (ACL'24) · [2402.10588](https://arxiv.org/abs/2402.10588) · ~183 (snippet). Three-phase depth trajectory through an English-biased "concept space."
- **1.5 The Hydra Effect: Emergent Self-repair** — McGrath et al. (DeepMind), 2023 · [2307.15771](https://arxiv.org/abs/2307.15771) · ~149 (S2 API). Downstream layers compensate for ablated earlier ones — depth computation is adaptive/self-correcting.
- **1.6 Analyzing Transformers in Embedding Space** — Dar et al., 2022 (ACL'23) · [2209.02535](https://arxiv.org/abs/2209.02535) · ~110 (snippet). Parameters (not only activations) interpretable via vocab-space projection.
- **1.7 Jump to Conclusions: Short-Cutting Transformers** — Yom Din et al., 2023 (LREC-COLING'24) · [2303.09435](https://arxiv.org/abs/2303.09435) · **101 (S2 API)**. Linear maps short-cut hidden states to final rep; models often reach prediction early.
- **1.8 Interpreting GPT: the Logit Lens** — nostalgebraist, 2020 · [blog](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) · N/A. The original logit lens. *Flagged: informal blog, not a paper.*

## Theme 2 — Token/time trajectories & dynamics (across the sequence)
Latent state tracking / world models; belief-state geometry; generation dynamics; attractors.

- **2.1 The Curious Case of Neural Text Degeneration** — Holtzman et al., 2019 (ICLR'20) · [1904.09751](https://arxiv.org/abs/1904.09751) · ~2,809 (mirror). Repetition loops under maximization decoding; introduces nucleus (top-p). Output-space, not activations.
- **2.2 Language Models Represent Space and Time** — Gurnee & Tegmark, 2023 (ICLR'24) · [2310.02207](https://arxiv.org/abs/2310.02207) · ~231 (snippet). Linear multi-scale space/time coordinates + neurons — world-model ingredients in activation geometry.
- **2.3 Emergent World Representations (Othello-GPT)** — Li et al., 2022 (ICLR'23 oral) · [2210.13382](https://arxiv.org/abs/2210.13382) · ~220 (mirror). Emergent, causally-editable board state evolving token-by-token.
- **2.4 Emergent Linear Representations in World Models** — Nanda et al., 2023 (BlackboxNLP) · [2309.00941](https://arxiv.org/abs/2309.00941) · ~104 (mirror). Othello world-model is encoded *linearly* ("mine vs opponent") → readable/steerable.
- **2.5 Transformers Represent Belief State Geometry in their Residual Stream** — Shai et al., 2024 (NeurIPS'24) · [2405.15943](https://arxiv.org/abs/2405.15943) · ~23 (snippet). Residual stream linearly encodes the Bayesian *belief state* (sometimes fractal); trajectory = meta-dynamics of belief updating. **Central to the "trajectory as dynamical object" view.**
- **2.6 Emergent World Models in Chess-Playing LMs** — Karvonen, 2024 (COLM'24) · [2403.15498](https://arxiv.org/abs/2403.15498) · ~11 (mirror). Probeable/editable board state; skill vector raises win rate up to 2.6×.
- **2.7 Unveiling Attractor Cycles… Successive Paraphrasing (Dynamical Systems View)** — Wang et al., 2025 · [2502.15208](https://arxiv.org/abs/2502.15208) · N/A. Iterated paraphrasing → stable 2-period attractor cycles. Explicit dynamical-systems framing of collapse.
- **2.8 Transformer Dynamics: A Neuroscientific Approach** — Fernando & Guitchounts, 2025 · [2502.12131](https://arxiv.org/abs/2502.12131) · ~1. Residual stream as dynamical system; cross-layer continuity, unstable periodic orbits, attractor-like dynamics (layer-wise more than token-wise).
- **2.9 Hallucination as Trajectory Commitment (Asymmetric Attractor Dynamics)** — Akarlar, 2026 · [2604.15400](https://arxiv.org/abs/2604.15400) · ~0. Hallucination = locally stable attractor basin (easy in, hard out). *Flagged: single-author, very recent.*
- **2.10 Attractor Geometry of Transformer Memory** — Liang, Miikkulainen & Fiete, 2026 — **WITHDRAWN** · [2605.05686](https://arxiv.org/abs/2605.05686). Facts as attractor basins. *Cite only with withdrawal caveat.*
- *Unverified leads (verify before citing): 2503.02854 "(How) Do LMs Track State?"; 2512.23722; 2603.16689.*

## Theme 3 — Representation engineering & activation-steering toolkit
*(Theme 3 counts are clean S2 API reads, 2026-09-07.)*

- **3.1 Inference-Time Intervention (ITI)** — K. Li et al., 2023 (NeurIPS'23 spotlight) · [2306.03341](https://arxiv.org/abs/2306.03341) · **1,435**. Shift attention-head activations along truth-correlated directions (TruthfulQA 32.5→65.1%).
- **3.2 Representation Engineering (RepE)** — Zou et al., 2023 · [2310.01405](https://arxiv.org/abs/2310.01405) · **1,315**. PCA "reading vectors" from contrastive stimuli, added/subtracted in the residual stream to monitor/control concepts.
- **3.3 Contrastive Activation Addition (CAA)** — Panickssery [né Rimsky] et al., 2023 (ACL'24) · [2312.06681](https://arxiv.org/abs/2312.06681) · **1,048**. Steering vector = mean activation difference of paired ±examples, added at post-prompt positions. *(This playground's method.)*
- **3.4 Activation Addition / ActAdd** — Turner et al., 2023 · [2308.10248](https://arxiv.org/abs/2308.10248) · **804**. Steering vector from a single contrast prompt pair; no optimization.
- **3.5 The Linear Representation Hypothesis & the Geometry of LLMs** — Park et al., 2023 (ICML'24) · [2311.03658](https://arxiv.org/abs/2311.03658) · **687**. Concepts as linear directions; causal inner product unifying embedding/unembedding. Theory for why linear steering works.
- **3.6 In-Context Learning Creates Task Vectors** — Hendel et al., 2023 (EMNLP'23 Findings) · [2310.15916](https://arxiv.org/abs/2310.15916) · **348**. ICL compresses demos into a task vector read from the last-demo hidden state.
- **3.7 Function Vectors in LLMs** — Todd et al., 2023 (ICLR'24) · [2310.15213](https://arxiv.org/abs/2310.15213) · **298**. A compact FV transported by a few heads triggers an in-context task when injected.
- **3.8 In-Context Vectors** — Liu et al., 2023 (ICML'24) · [2311.06668](https://arxiv.org/abs/2311.06668) · **286**. Latent-space shift from demos, added to query hidden states with tunable magnitude.

## Theme 4 — Feature bases for interpreting trajectories
SAEs / dictionary learning, superposition, residual-stream framework.

- **4.1 Toy Models of Superposition** — Elhage et al. (Anthropic), 2022 · [transformer-circuits](https://transformer-circuits.pub/2022/toy_model/index.html) / [2209.10652](https://arxiv.org/abs/2209.10652) · ~527 (snippet). Why features exceed dimensions along non-orthogonal directions → need overcomplete dictionaries.
- **4.2 SAEs Find Highly Interpretable Features** — Cunningham et al., 2023 (ICLR'24) · [2309.08600](https://arxiv.org/abs/2309.08600) · 107. Overcomplete dictionary of sparse, largely monosemantic residual-stream features.
- **4.3 Scaling and Evaluating Sparse Autoencoders** — Gao et al. (OpenAI), 2024 · [2406.04093](https://arxiv.org/abs/2406.04093) · 102. TopK SAEs, clean scaling laws, up to 16M features.
- **4.4 Sparse Feature Circuits** — Marks et al., 2024 (ICLR'25) · [2403.19647](https://arxiv.org/abs/2403.19647) · 56. Causal circuits from SAE features → editable interpretable graphs.
- **4.5 Gated Sparse Autoencoders** — Rajamanoharan et al. (DeepMind), 2024 · [2404.16014](https://arxiv.org/abs/2404.16014) · 23. Decouples detection from magnitude; removes L1 shrinkage bias.
- **4.6 Towards Monosemanticity** — Bricken et al. (Anthropic), 2023 · [transformer-circuits](https://transformer-circuits.pub/2023/monosemantic-features/index.html) · N/A (no arXiv). SAE on a 1-layer model → thousands of monosemantic features.
- **4.7 Scaling Monosemanticity (Claude 3 Sonnet)** — Templeton et al. (Anthropic), 2024 · [transformer-circuits](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) · N/A. ~34M steerable features at production scale.
- **4.8 A Mathematical Framework for Transformer Circuits** — Elhage et al. (Anthropic), 2021 · [transformer-circuits](https://transformer-circuits.pub/2021/framework/index.html) · N/A (no arXiv). Residual-stream-as-communication-channel; read/write subspaces, QK/OV, induction heads.

## Theme 5 — Methods for analyzing trajectories
Probing across layers/time; dimensionality reduction; geometry / intrinsic dimension; state-space models.

- **5.1 Understanding Intermediate Layers Using Linear Classifier Probes** — Alain & Bengio, 2016 (ICLR'17 WS) · [1610.01644](https://arxiv.org/abs/1610.01644) · ~1,013 (snippet). Per-layer linear probes — foundational cross-layer method.
- **5.2 Probing Classifiers: Promises, Shortcomings, Advances** — Belinkov, 2021/22 (*Comp. Ling.* 48(1)) · [2102.12452](https://arxiv.org/abs/2102.12452) · ~559 (snippet). Methodological survey of probing (pitfalls incl. the position/decodability confounds).
- **5.3 Intrinsic Dimension of Data Representations in DNNs** — Ansuini et al., 2019 (NeurIPS'19) · [1905.12784](https://arxiv.org/abs/1905.12784) · ~535 (snippet). TwoNN ID across layers → "hunchback" profile (expand then contract).
- **5.4 The Geometry of Hidden Representations of Large Transformer Models** — Valeriani et al., 2023 (NeurIPS'23) · [2302.00294](https://arxiv.org/abs/2302.00294) · N/A. ID + neighbor-overlap across layers; ID minima flag semantically-richest layers. Best match for "ID of the hidden-state trajectory."
- **5.5 Geometry of Categorical and Hierarchical Concepts** — Park et al., 2024 (ICLR'25 oral) · [2406.01506](https://arxiv.org/abs/2406.01506) · N/A. Concepts as simplices/polytopes; hierarchy as orthogonality. *Static geometry, not temporal.*
- **5.6 Visualizing LLM Latent Space Geometry Through Dimensionality Reduction** — Ning et al., 2025 · [2511.21594](https://arxiv.org/abs/2511.21594) · N/A. PCA/UMAP of latent states; attention/MLP separation, high-norm initial states, helical positional structure.
- **5.7 Markovian Circuit Tracing for Transformer State Dynamics** — "Abdullah X", 2026 · [2605.20824](https://arxiv.org/abs/2605.20824) · N/A. Discretize activations → transition matrices, Markov-order tests, state-forcing. *Flagged: single independent author, non-peer-reviewed.*
- *Off-target: Torre et al., NeurIPS'25, [2510.25674](https://arxiv.org/abs/2510.25674) — RNNs, not transformer hidden states.*

---

## Synthesis & gaps

**Well-studied:**
- **Depth (layer-wise) trajectories are mature** — logit lens → tuned lens → short-cutting / DoLa (Theme 1), grounded in "iterative inference" and the residual-stream framework; self-repair (Hydra) shows the depth trajectory is adaptive.
- **Static steering vectors are crowded & high-impact** — ITI (~1,435), RepE (~1,315), CAA (~1,048), ActAdd (~804), plus function/task/in-context vectors. Dominant paradigm: one linear direction from contrastive activations, added at inference. Linear representation hypothesis supplies the theory.
- **Feature bases maturing** — SAEs/dictionary learning (Theme 4) scale to tens of millions of features and connect to editable circuits.
- **Cross-sectional geometry & probing** (Theme 5) well-established layer-by-layer.

**Under-studied (the frontier):**
- **Generation-time (token/time) trajectory as a first-class dynamical object.** Most work is *static* (a vector at a position/layer), not a *path* over autoregressive steps. World-model line (Othello/Chess) shows latent state updates token-by-token; belief-state geometry (Shai 2024) is the strongest account of *what* it represents — but not the *dynamics of the trajectory itself*.
- **Convergence & attractor dynamics** — a genuine dynamical-systems treatment (fixed points, basins, cycles, stability) is only emerging in 2025–26 preprints (paraphrase attractor cycles; hallucination-as-attractor; a withdrawn attractor-geometry paper; Transformer Dynamics), thin and often non-archival. Classic degeneration (Holtzman) named it in *output* space, never *activation* space.
- **State-space / HMM models of hidden-state evolution** — nearly absent (only a caveated Markov preprint; RNN-not-transformer work).
- **Trajectory-aware / closed-loop steering** — the whole Theme 3 toolkit intervenes *statically*; little on steering that accounts for how the intervention propagates through subsequent dynamics.

**Bottom line:** strong tools for the *depth* trajectory (lenses), the *feature basis* (SAEs), and *static* linear steering — but treating the *generation-time activation path* as a trajectory / dynamical object, with convergence and attractor structure, is a genuine, fast-moving frontier where evidence is still preliminary. (This is exactly where this playground's degeneration/convergence experiments sit.)

### Flagged
- 2.9, 2.10, 5.7 — very recent / single-author / non-archival; **2.10 (2605.05686) withdrawn**.
- 1.8 logit lens — informal blog, no arXiv.
- Theme-2 unverified leads (2503.02854, 2512.23722, 2603.16689) — verify before citing.
- Counts: Theme 3 = direct S2 API; others = snippets/mirrors, approximate and time-varying.
