#!/usr/bin/env python3
"""
Label the nftables article spans with security/vulnerability classification.
CVE-2024-1086: Linux kernel netfilter nftables double-free vulnerability
"""

import json

# Initialize labels dictionary
L = {}

# Spans 0-15 (Lines 1-16): Blog header/navigation/furniture
for i in range(0, 16):
    L[str(i)] = "background"

# Spans 16-37 (Lines 17-38): Introduction, abstract, abstract overview
for i in range(16, 38):
    L[str(i)] = "background"

# Spans 38-44 (Lines 39-45): Section 1 "Overview" begins, and early content
for i in range(38, 45):
    L[str(i)] = "background"

# Spans 45-72 (Lines 46-73): Section 2 "Background info" sections 2.1-2.3
# nf_tables, sk_buff, IP packet fragmentation - all technical background
for i in range(45, 73):
    L[str(i)] = "background"

# Spans 73-146 (Lines 74-147): Section 3.1 "Finding the bug" and 3.2 "Root cause analysis"
# This section shows vulnerable kernel code, investigation/analysis process
# 3.1 is DEBUGGING (investigation), 3.2 is UAF (mechanism in the code)
for i in range(73, 107):
    L[str(i)] = "debugging"

for i in range(107, 147):
    L[str(i)] = "uaf"

# Spans 147-177 (Lines 148-178): Section 3.3 "Bug impact & exploitation" and 3.4 "Bug fixes"
# 3.3 describes the double-free impact (still mechanism-focused) - UAF
# 3.4 shows the CVE description and patch code - PATCH
for i in range(147, 162):
    L[str(i)] = "uaf"

for i in range(162, 178):
    L[str(i)] = "patch"

# Spans 178-195 (Lines 179-196): Section 4 "Techniques" begins, 4.1 "Page refcount juggling"
# Exploitation technique for enabling double-free via page refcount juggling
for i in range(178, 196):
    L[str(i)] = "exploit"

# Spans 196-212 (Lines 197-213): Section 4.2 "Page freelist entry order 4 to order 0"
# PCP draining technique and obsolete race condition technique
for i in range(196, 213):
    L[str(i)] = "exploit"

# Spans 213-222 (Lines 214-223): Section 4.3 "Freeing skb instantly"
# and Section 4.3.1 "Modifying skb max lifetime"
# Exploitation using IP fragmentation queues
for i in range(213, 223):
    L[str(i)] = "exploit"

# Spans 223-252 (Lines 224-253): Section 4.4 "Bypassing KernelCTF skb corruption checks"
# Technique for bypassing freelist corruption detection
for i in range(223, 253):
    L[str(i)] = "exploit"

# Spans 253-266 (Lines 254-267): Section 4.5 "Dirty Pagedirectory"
# Detailed explanation of the Dirty Pagedirectory technique
for i in range(253, 267):
    L[str(i)] = "exploit"

# Spans 267-350 (Lines 268+): Section 4.6, 4.7, 4.8 - Spraying, TLB flushing, KASLR defeat
# All exploitation techniques for setting up the primitive
for i in range(267, 400):
    L[str(i)] = "exploit"

# Spans 400-550: Continued section 4 - more techniques and PoC setup
for i in range(400, 551):
    L[str(i)] = "exploit"

# Spans 551-750: Section 5 "Proof of Concept" - exploit execution and PoC code
for i in range(551, 751):
    L[str(i)] = "exploit"

# Spans 751-800: Section 5.3 "Build instructions" and related build code
# Still part of PoC (exploit compilation/execution)
for i in range(751, 801):
    L[str(i)] = "exploit"

# Spans 801-817 (Lines 802-818): Footer - related articles, blog signature, navigation
for i in range(801, 818):
    L[str(i)] = "background"

# Verify we have labeled all 818 spans
assert len(L) == 818, f"Expected 818 labels, got {len(L)}"
assert all(0 <= int(k) < 818 for k in L.keys()), "Invalid span indices"

# Build the output JSON
output = {
    "article": "nftables",
    "labeller": "haiku",
    "n_spans": 818,
    "labels": {str(i): L[str(i)] for i in range(818)}
}

# Verify all labels are from vocabulary
valid_labels = {
    "uaf", "heap_overflow", "stack_overflow", "integer_overflow",
    "race_condition", "oob_access", "command_injection", "path_traversal",
    "auth_bypass", "priv_esc", "uninit_memory", "other_vuln",
    "background", "debugging", "exploit", "patch", "benign"
}

for label in output["labels"].values():
    assert label in valid_labels, f"Invalid label: {label}"

print(json.dumps(output, indent=2))
