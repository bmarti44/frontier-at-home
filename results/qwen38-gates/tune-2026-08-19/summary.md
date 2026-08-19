# Qwen3.8-27B speed-tune gate — 2026-08-19 (Track A2, code-focused)

Q4_K_M, b10488, port 8015, capped unit, one production window.

## Phase P — micro-batch prefill sweep: NULL RESULT (preserved)

Strict 28K bench cells (MTP n=2 fixed):

| ub (batch) | decode | prefill (incl. setup) |
|---|---|---|
| 512 (b 2048, default) | 18.30 | **699** |
| 2048 (b 4096) | 18.50 | 687 |
| 4096 (b 4096) | 18.27 | 567 |

Unlike vanilla-transformer GB10 results (community -ub 2048 standard),
this hybrid GDN architecture gains nothing from larger micro-batches —
the chunked DeltaNet prefill path does not reward them, and ub 4096 is
strictly worse. Default -b 2048 -ub 512 stays.

## Phase D — MTP draft sweep, scored by greedy code probes

Two probes per config via /completion server timings: "gen" (write LRU
cache + tests), "refactor" (refactor 6KB of real repo Python). tok/s:

| config | gen | refactor | avg |
|---|---|---|---|
| n2 p0 (prior default) | 23.56 | 22.16 | 22.9 |
| n3 p0 | 26.72 | 26.74 | 26.7 |
| n4 p0 | 28.69 | 23.95 | 26.3 |
| n4 p0.6 | 27.89 | 25.02 | 26.5 |
| **n8 p0.6 (winner)** | **27.26** | **27.80** | **27.5** |
| n8 p0.8 | 27.00 | 24.37 | 25.7 |
| n16 p0.8 | 25.90 | 22.78 | 24.3 |

Winner: `--spec-draft-n-max 8 --spec-draft-p-min 0.6` — +20% over the
n2 baseline on code, best worst-case. The 0.6 confidence gate makes
depth pay on this bandwidth-limited part (0.8 over-prunes; ungated n4+
wastes rejected drafts on less predictable content). Matches community
guidance that gated-deep beats shallow, with the GB10-specific optimum
at a softer gate than desktop Blackwell.

Probes are supplementary evidence (greedy, fixed prompts, server
timings); README cells come only from 30_bench strict runs. Validation
window (canary, MTP on/off greedy equivalence at n8p6, strict 0-ctx
cell) follows in validate_n8p6.
