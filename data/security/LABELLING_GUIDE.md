# CVE write-up span labelling guide

Every span in an article gets **exactly one** label. Spans are produced by
`scripts/reft/span_labels.py` and the indices are stable — never renumber them.

## Mechanism labels — use ONLY when the span actually describes the bug's mechanism

| label | use when the span describes… |
|---|---|
| `uaf` | use-after-free, double-free, dangling pointer, refcount error, stale object reuse |
| `heap_overflow` | heap buffer overflow/underflow, slab corruption, out-of-bounds heap write |
| `stack_overflow` | stack buffer overflow, saved-return-address smash |
| `integer_overflow` | integer overflow/underflow/truncation/sign confusion leading to a bad size |
| `race_condition` | TOCTOU, missing lock, concurrent access, CoW race, signal race |
| `oob_access` | out-of-bounds read/write that is not specifically heap or stack (index, argv, array) |
| `command_injection` | injected code/command/expression evaluated by an interpreter (shell, SpEL, OGNL, eval) |
| `auth_bypass` | authentication/authorisation/credential check defeated |
| `priv_esc` | the privilege transition itself (uid→0, token steal, cred struct overwrite) |
| `uninit_memory` | uninitialised variable/struct/flag being read or inherited |
| `other_vuln` | a real bug mechanism that fits none of the above |

A mechanism label means the span **explains or demonstrates that flaw**, not that it merely
mentions the CVE name. "We found a use-after-free in nftables" in an intro paragraph is
`background`; the paragraph explaining *which* refcount path drops early is `uaf`.

## Narrative labels — everything else

| label | use when |
|---|---|
| `background` | intro, setup, versions, environment, target description, history, motivation, conclusions, acknowledgements, tables of contents, references |
| `debugging` | crash dumps, gdb/WinDbg sessions, KASAN/ASAN reports, logs, instrumentation, traces, "we added a printk" |
| `exploit` | exploitation strategy and exploit code: heap grooming, spraying, ROP, primitive construction, the PoC itself |
| `patch` | the upstream fix, patch hunks, commit messages, mitigation advice, "the fix adds a bounds check" |
| `benign` | source code, prose or output with no security content — correct code being shown for reference, boilerplate, build commands |

## Rules that decide the hard cases

1. **Diff hunks.** In a patch listing, `-` lines (the code being removed) take the *mechanism*
   label — that is the vulnerable code. `+` lines (the fix) are `benign`. Context lines and
   the surrounding commit prose are `patch`.
2. **Environment facts** ("Ubuntu 20.04", "kernel 5.11.0-44", "tested on iOS 14.3") are
   `background`, even when they appear inside an exploit section.
3. **Code flag is separate from the label.** A fenced block that *is* the exploit is
   `exploit`; a fenced block that is the vulnerable kernel source is the mechanism label;
   a fenced block of gdb output is `debugging`.
4. **Headings** take the label of the section they open (`## The bug` → the mechanism if
   the section explains it, otherwise `background`).
5. **Prefer the narrative label when unsure.** Mechanism spans are rare — roughly 5% of a
   write-up. Over-applying them is the main failure mode. If a paragraph is mostly about
   *how the author exploited* the flaw rather than *what the flaw is*, it is `exploit`.
6. One article usually has **one dominant mechanism**. Do not sprinkle five different
   mechanism classes through it; only label a second mechanism if the text genuinely
   describes a second distinct flaw.

## Output format

Write `data/security/span_labels/{article}.json`:

```json
{"article": "<name>", "labeller": "haiku-agent", "n_spans": <int>,
 "labels": {"0": "background", "1": "background", "2": "uaf", ...}}
```

`labels` must contain an entry for **every** span index from `0` to `n_spans-1`.
