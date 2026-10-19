# Article concept audit — brief

We are rebuilding the CVE corpus around a much simpler, 3-way scheme. For each article there is
exactly ONE primary security concept, and the rest of the article is either preamble or the
investigation narrative. Your job is to read an article and report which concept it is, and
roughly how the article divides up.

## The three label roles (for your % estimate)

- **background** — preamble. What the target software is, environment and versions, history,
  prior work, related CVEs, timeline, references, acknowledgements, conclusions. Also generic
  explanations of subsystems that are not the bug itself (how the page allocator works, what a
  Mach port is).
- **triage** — the investigation narrative: *what was tried*. Fuzzing, crash reproduction,
  debugger sessions, sanitizer output, patch diffing, hypothesis testing, and ALSO the
  exploitation work (heap grooming, building primitives, bypasses, the PoC itself).
- **mechanism** — the text that actually explains the flaw: which line is wrong, why the check
  is insufficient, what the incorrect state transition is, the vulnerable code and its
  explanation.

Mechanism is normally the SMALLEST of the three (often 3-10% of an article). Do not inflate it.

## Concept vocabulary — pick the PRIMARY one

`uaf` (use-after-free, double-free, refcount/dangling-pointer error)
`heap_overflow` (heap buffer overflow/underflow, slab or pool corruption)
`stack_overflow` (stack buffer overflow)
`integer_overflow` (integer overflow/underflow/truncation/sign error producing a bad size)
`oob_access` (out-of-bounds index/read/write that is not specifically heap or stack)
`race_condition` (TOCTOU, missing lock, concurrent access, CoW race, signal race)
`command_injection` (injected command/code/expression evaluated by an interpreter)
`uninit_memory` (uninitialised variable/struct/flag read or inherited)
`auth_bypass` (authentication/authorisation check defeated)
`other` (a real mechanism fitting none of the above — say what it is)

Judge by what the article spends its MECHANISM text on. If a bug is "an integer underflow that
leads to a heap overflow", pick whichever the author actually explains as the root cause, and put
the other in `secondary`.

## Output

Write a single JSON file at the path you are given:

```json
{"<article>": {"primary": "uaf", "secondary": "race_condition", "confidence": "high",
               "pct_background": 45, "pct_triage": 48, "pct_mechanism": 7,
               "root_cause_one_line": "...",
               "evidence": ["short verbatim quote that is the clearest mechanism sentence",
                            "another one"]},
 "<next article>": {...}}
```

`secondary` may be `null`. `confidence` is `high` / `medium` / `low`. The three percentages
should total 100. Give 2-3 short verbatim `evidence` quotes (one sentence each) taken from the
mechanism text — these are how we check your call, so quote exactly.
