# Security LoReFTs (layer 17, rank 4)

Trained 11 concepts x 2 position schemes on data/security/paragraphs_security.json.
Prompt: component description (no vulnerability requested) -> the concept's write-up. 3 held out per concept.
Peak VRAM 9.60 GB, 1138s total. wd=0, gradient checkpointing above L17.

Sanity check: mean layer-17 DiM score for the TARGET concept on held-out prompts (plain model -> LoReFT; corpus value in parens),
and how often the LoReFT output's nearest of the 10 vulnerability classes is the target.

| concept | scheme | best epoch | held loss | target-DiM plain | -> LoReFT | corpus | nearest=target |
|---|---|---|---|---|---|---|---|
| uaf | all | 6 | 1.187 | +0.01 | +0.82 | +1.36 | 0.667 |
| uaf | last | 12 | 1.211 | +0.01 | +1.25 | +1.36 | 1.0 |
| heap_overflow | all | 6 | 1.218 | +0.35 | +0.78 | +1.11 | 0.333 |
| heap_overflow | last | 7 | 1.282 | +0.35 | +0.45 | +1.11 | 0.0 |
| stack_overflow | all | 12 | 0.930 | -0.36 | +0.56 | +1.12 | 0.667 |
| stack_overflow | last | 10 | 0.978 | -0.36 | +0.48 | +1.12 | 0.667 |
| integer_overflow | all | 9 | 0.869 | +0.13 | +0.69 | +1.41 | 0.333 |
| integer_overflow | last | 11 | 0.908 | +0.13 | +0.56 | +1.41 | 0.667 |
| race_condition | all | 4 | 1.020 | +0.90 | +1.22 | +1.54 | 0.667 |
| race_condition | last | 12 | 1.013 | +0.90 | +1.25 | +1.54 | 0.667 |
| oob_access | all | 9 | 0.925 | +1.16 | +1.51 | +1.39 | 0.667 |
| oob_access | last | 9 | 0.922 | +1.16 | +1.25 | +1.39 | 0.667 |
| command_injection | all | 6 | 0.942 | +0.38 | +1.18 | +1.54 | 1.0 |
| command_injection | last | 12 | 1.019 | +0.38 | +1.17 | +1.54 | 0.667 |
| path_traversal | all | 7 | 1.094 | +0.46 | +1.31 | +1.49 | 1.0 |
| path_traversal | last | 12 | 1.132 | +0.46 | +0.79 | +1.49 | 0.333 |
| auth_bypass | all | 6 | 1.292 | +1.45 | +1.57 | +1.72 | 0.667 |
| auth_bypass | last | 6 | 1.264 | +1.45 | +1.75 | +1.72 | 1.0 |
| priv_esc | all | 5 | 1.547 | +1.06 | +1.42 | +1.34 | 0.333 |
| priv_esc | last | 6 | 1.554 | +1.06 | +1.38 | +1.34 | 0.333 |
| benign | all | 3 | 1.290 | +2.08 | +1.41 | +1.48 | None |
| benign | last | 5 | 1.298 | +2.08 | +1.37 | +1.48 | None |
