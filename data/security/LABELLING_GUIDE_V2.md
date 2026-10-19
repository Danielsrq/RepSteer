# Labelling guide — the 5-label scheme (Design B)

Every span gets exactly ONE of **five** labels:

    background   triage   overflow   uaf   race

Unlike the earlier version, an article may carry **more than one** security concept. You will be
told the article's PRIMARY concept, but if the text genuinely discusses another concept's
mechanics you may use that label too. The three concept labels are:

- **overflow** — buffer/heap/stack overflow, integer overflow/underflow, out-of-bounds read or
  write. Includes the OOB write/read itself AND reasoning about **what the overflow can reach or
  overwrite**.
- **uaf** — heap-object **lifetime** manipulation: use-after-free, double-free, and also the
  grooming that characterises these bugs — allocating, freeing and reusing objects, heap
  spraying, slab/freelist shaping, cross-cache, unlink/arbitrary-free primitives, overlapping a
  freed chunk, heap/pointer leaks obtained by shaping the heap. **`uaf` here means "heap lifetime",
  not strictly free-then-use** — spray and slab shaping count.
- **race** — a race condition or TOCTOU: the timing window itself and how it is won (e.g. using
  FUSE to stall a copy, racing `madvise` against `write`).

The point of Design B: label the concept by the **security area the text is about**, not only by
the one sentence that states the root cause. Exploitation prose is full of concept-bearing content
— a paragraph about spraying `msg_msg` objects and freeing them to overlap a `pipe_buffer` IS
about heap lifetime, so it is `uaf`, even though it is "exploitation".

---

## The three roles

### `background` — preamble AND how the system normally works

- Title, author, CVE number, table of contents, section banners, conclusions, acknowledgements,
  timeline, footnotes, references, and blog-comment / social / footer junk ("ReplyDelete",
  "Subscribe to: Post Comments", "Posted by …").
- What the target software is, versions, environment, history, prior CVEs, impact.
- **Any explanation of how the subsystem NORMALLY works**, however long and however much kernel
  source it contains. If the text teaches how something is *supposed* to behave and it is neither
  the defect nor a concept operation nor something the author did, it is `background`.
- **Reference source listings** — code shown to follow along, which is not the defective code and
  not a concept operation.

### `triage` — class-agnostic exploitation plumbing and investigation

`triage` is what is left of "what was tried / done" once the concept-specific parts are pulled
out. Use it for work that is **not specific to the vulnerability class**:

- Finding the bug: fuzzing, crash reproduction, debugger/sanitizer sessions, patch diffing.
- The **patch / fix** and its diff (there is no `patch` label; do not split a diff by +/-).
- Generic exploitation infrastructure: ROP chains, gadget hunting, stack pivots, control-flow
  hijack, target selection (e.g. `modprobe_path`), address brute-forcing, environment quirks,
  privilege-escalation payloads, flag dumping.

### `overflow` / `uaf` / `race` — the concept, wherever it appears

- The defective code and the prose explaining why it is wrong.
- **The operations characteristic of that vulnerability class**, including in the exploitation
  section: for `overflow`, the OOB write/read and what it overwrites/reaches; for `uaf`, all the
  heap alloc/free/spray/slab/unlink/overlap/leak shaping; for `race`, the timing window and how it
  is won.

Deciding between `triage` and a concept: ask *"is this text specifically about the mechanics of a
memory-safety class, or is it generic exploit plumbing?"* Heap grooming → `uaf`. ROP gadgets →
`triage`. The overflow's reach → `overflow`. Brute-forcing a stack address → `triage`.

---

## Rules that decide the hard cases

1. **Naming the bug is `background`; explaining or operating on it is the concept.**
   "We found a heap overflow via fuzzing" → the fuzzing is `triage`, the phrase names the class.
   The paragraph explaining *how* the overflow happens, or *what it overwrites*, is `overflow`.

2. **A concept can appear anywhere, including deep in the exploitation section.** Label by what the
   span is about, never by which section it sits in.

3. **Normal behaviour of the vulnerable subsystem is `background`, however technical.** A long
   walkthrough of kernel internals (allocators, page faults, `mem_rw`, `__get_user_pages`) is
   `background` when it describes how things normally work.

4. **Heap grooming is `uaf`, control-flow hijack is `triage`.** Spraying, freeing, slab shaping,
   unlink, overlapping a freed chunk, heap leaks → `uaf`. ROP/gadgets/pivots/target-selection →
   `triage`. This is the seam that matters most under Design B — get it right.

5. **The patch section is `triage`**, diff included.

6. **Headings take the label of the section they open.** Decorative banners and TOC → `background`.

7. **Contentless separators** (`---`, `* * *`) inherit the surrounding label.

8. **Environment / version facts are `background`**, even inside exploitation.

---

## Worked example: cve_2022_0185 (primary overflow; also uaf + race, 154 spans)

This is the reference example — it shows all three concepts co-occurring in one article.

    0-2      background   header, intro
    3-11     triage       finding: syzkaller fuzzing, KASAN crash log, C repro
    12-18    background   normal fsconfig / fs_context / legacy_parse_param behaviour
    19-22    overflow     the vulnerable function, the bad bound check, the integer underflow
    23-24    background
    25-26    triage       the patch and its diff
    27       triage       targeting Ubuntu 20.04
    28       overflow     "only a heap overflow" — reasoning about the overflow primitive
    29       triage       note on exploit stability
    30-35    uaf          spraying, KASLR-leak-via-chunk, MSG_COPY no-unlink, arb-write technique
    36-37    triage       FUSE technique intro + evil_read code
    38       race         "difficulties with this FUSE RACE technique"
    39       triage       targeting modprobe_path
    40       race         "use FUSE to time the race"; msg_msg arb write won via the race
    41-43    triage       exploit link, kCTF intro, barebones /dev environment
    44-46    uaf          GFP_KERNEL_ACCOUNT object placement, cross-cache overflow of msg_msg
    47       race         limited race control → choice of unlink / arbitrary-free primitive
    48-53    uaf          heap spray technique, "Heap Setup:", slab-hole shaping
    54       overflow     "Overflow into m_ts (size) to achieve OOB read"
    55-66    uaf          selective free/LIFO, heap-leak code, pipe_buffer-via-heap, unlink attack,
                          DEBUG_LIST check, linked-list overwrite, freeing+spraying to overlap
    67-77    triage       stack pivot, ROP gadgets, ROP chain, privesc, flag dump, conclusion
    78-153   background   blog comment section / footer junk

Result: background 89, uaf 27, triage 29, overflow 6, race 3.

## Worked example: looney_tunables (overflow, 58 spans)

A data-only overflow, so much of the "exploitation" IS the overflow's reach → `overflow`.

    0-11     background   advisory header, TOC, summary, normal parse_tunables behaviour
    12-16    overflow     the vulnerable function + the 3-step overflow explanation
    17       triage       fuzzing
    19       triage       the reproducer
    21-26    overflow     what the overflow can overwrite; the mmap buffer; reach analysis;
                          the mmap-two-and-overflow-one layout  (with 24 = maps listing = background)
    27-30    triage       what to replace with (LD_PRELOAD / dangerous-tunable dead ends)
    32,34,36 triage       calloc-init setup; failed l_next/l_prev attempt; DT_RPATH stood out
    33       overflow     the overflow overwrites link_map pointers with non-zero bytes
    38-43    triage       DT_RPATH target selection + stack address brute-force
    45-46    overflow     overflow-with-null-bytes problem; OOB bytes are also READ out of bounds
    47-50    triage       final notes, which SUID programs
    (18,20,24,31,35,37,39,44,51-57 background)

Result: background 26, overflow 13, triage 19.

## Worked example: dirtycow (race, 125 spans) — the "clean" case

A race with essentially no heap grooming, so Design B ≈ the defect-only labelling: the concept is
concentrated in the mechanism prose, and the thin exploitation (PoC threads, patch) stays `triage`.
Not every article gains concept spans in exploitation — only add them where the text is genuinely
about the class's mechanics.

    Result: background 79, race 28, triage 18.

---

## Output

Write `data/security/span_labels_v2/{article}.json`:

```json
{"article": "<name>", "labeller": "haiku-agent", "n_spans": <int>, "concept": "<primary concept>",
 "concepts_present": ["overflow", "uaf"], "labels": {"0": "background", ...}}
```

`labels` must contain an entry for **every** span index `0`..`n_spans-1`, each value one of
`background`, `triage`, `overflow`, `uaf`, `race`. `concepts_present` lists the concept labels you
actually used.
