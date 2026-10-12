# Candidate 10 result: PASS

The frozen scorer (`gate_verdict.py`, freeze `176e820f`, sha256 prefix
`85aa149cceb9e2be`) returns **PASS** in `gate-summary.json`, with no failed
clause, no missing evidence and no `GATE_*` override.
- Before scoring, `gate_mutations.py` matched **103/103**: 91 cases plus 12
  `--run2-preflight` cases.
- Run 1 ran at head `d7bc6a45` and run 2 at head `ff15f1f6`. Both were clean,
  and their FROZEN diff against the freeze was empty.
- The block order comes from drand round 6544802, the first round after the
  freeze commit's committer time: BAAB, ABBA, BAAB, ABBA, BAAB.

| Clause | Result |
|---|---|
| Kill lines (kernel, server, memwatch; run 1 and every run-2 start) | none |
| Swap growth | 0.0 GiB |
| Low point ≥ 8 GiB | run 1: 9.84 GiB, during the 1M fill. Run 2: 10.32–10.60 GiB over 20 starts |
| Direct 1M | KV capacity 1,079,117 at 262,144; fill PASS, 1,000,672 tokens; cap 1,048,576 |
| Text path vs candidate 5 | 3/3 greedy sha256 identical, so accuracy carries over and no holdout spend was needed |
| Tool-call | 20/20 |
| Agent gate | 4/4 |
| Soak at the 5 GiB repository floor | PASS; `mem_floor_gib` 5.0 from `default`; minimum 10.03 GiB; decode median 37.62 tok/s (candidate 9: 30.87) |
| Vision | 64/100, 3 invalid, 0 errors; z = 0.0 against 64 |
| Admission | peak 105.74 GiB (launch 115.58 − low 9.84); +8 = 113.74 ≤ 115 |
| Precondition | 17.41 GiB of qwen38-1m's GGUF resident at launch, plus 2.99 GiB of runtime-001 |
| Fixed worker | one release under `/home/bmarti44/models` (496 files); MemFree 21.82 GiB at the 17.90 GiB arena + 2 GiB. One release of `~/.cache/qwen38-flash-next` (136,225 files) |
| Runtime identity | every tree line equals the runtime-002 manifest: 001 `c08e99825f3704d3`/5352, 002 `9e2e0f3138668a75`/5353 |
| Lever identity | every candidate start logs L1 (PLE WILLNEED) and L2 (draft vocab 65,536, NVFP4 draft head) |
| TTFT at 0 | 0.21 s ≤ 0.5 s |

## Run 2: matched speed (5 fresh-start blocks, B = candidate 10, A = candidate 9)

All 20 starts recorded `failed=None kills=0`. Run 2 took 04:17:47Z to
09:09:58Z.

| Metric | B/A ratio | One-sided 95% lower | Upper | Rule | Status |
|---|---|---|---|---|---|
| Decode at 0 | **1.385** | 1.352 | 1.420 | lower ≥ 1.0 and point ≥ 1.10 | PASS |
| Decode at 28,672 | **1.189** | 1.172 | 1.206 | lower ≥ 1.0 | PASS |
| Prefill at 28,672 | 1.010 | 1.008 | 1.012 | lower ≥ 0.95 | PASS |
| TTFT at 0 | 0.730 | 0.695 | 0.768 | reported | — |

Per-block arm means (A → B, tok/s):

| Block | Order | Decode at 0 | Decode at 28,672 | Prefill at 28,672 |
|---|---|---|---|---|
| 0 | BAAB | 23.51 → 34.00 | 35.10 → 41.66 | 2,672 → 2,700 |
| 1 | ABBA | 24.56 → 34.10 | 34.76 → 41.93 | 2,671 → 2,700 |
| 2 | BAAB | 24.82 → 34.05 | 35.83 → 43.17 | 2,673 → 2,702 |
| 3 | ABBA | 24.90 → 34.01 | 35.43 → 41.25 | 2,683 → 2,702 |
| 4 | BAAB | 24.99 → 33.88 | 34.77 → 41.08 | 2,686 → 2,715 |

These are fresh-start, cold-PLE runs on English prose (the speed fixture).

## Reported, not gating

**The kit's numbers, from `SUMMARY.md` (kit verdict PASS):**
- **Decode:** 34.07 tok/s at 0 and 40.92 at 28,672. Candidate 9 measured
  25.76 and 34.89.
- **Prefill:** 2,711.49 tok/s at 28,672.
- **TTFT:** 0.21 s at 0 and 12.80 s at 28,672.

**Decode on CJK output** (`cjk_decode.py`; single start; diagnostic only).
L2's draft proposes only ids below 65,536, and most Chinese and Japanese
output ids lie at or above that. Each engine is compared by its ratio to its
own English decode, as preregistered:

| Language | Candidate 9 (production) | Ratio to English | Candidate 10 (run 1) | Ratio to English | Output ids ≥ 65,536 |
|---|---|---|---|---|---|
| English | 25.67 tok/s | 1 | 34.46 tok/s | 1 | 4–5% |
| Chinese | 22.27 tok/s | 0.87 | 18.21 tok/s | 0.53 | 79% |
| Japanese | 19.56 tok/s | 0.76 | 20.16 tok/s | 0.59 | 68–69% |

- English is about 1.34× faster.
- Japanese is about level, at 1.03×.
- **Chinese is about 18% slower** in absolute tok/s, at 0.82×.

Candidate 9's numbers were taken on live production (warm PLE, possibly with
concurrent traffic) and candidate 10's at the end of run 1, so the absolute
cross-engine figures are indicative only.

**Fidelity is `NO_RESULT`, because there is no BF16 teacher on this host.**
Losslessness rests on:
- clause 5's three byte-identical greedy probes;
- the direct-1M retrieval;
- the tool-call and agent gates;
- the screen's byte-identical probes in every runtime-002 arm.

The PREREG's "Reported, not gating" section names L0's target-path code.

## Convergence note for the owner (reviewer 2)

**Blocking findings by candidate were 0, 1, 0.** Round 2's single HIGH:
- was introduced by round 1's coverage fix;
- failed closed (`NO_RESULT`), not as a false PASS;
- was closed in candidate 3.

**Under the reading recorded in PREREG, the gate did not pause.** AGENTS.md's
pause clause, "blocking findings fail to strictly decrease across two
consecutive candidates", was read as two non-decreases in a row.

**Reviewer 2 notes another reading.** The literal pause clause together with
continue-condition 3 ("no new finding class") could have paused the gate
after candidate 2. The owner decides whether that changes anything.

## Host

**The run-2 driver released the runtime page cache at exit (09:10:05Z).**
- No scope remained.
- Port 8017 and the inference lock were free.
- MemAvailable was 114.75 GiB.

**Production was restored with `52_engine_switch.sh restore` at 09:10:22Z.**
- `status --json` reports `qwen38fn`.
- `:8013` serves, and the switch's semantic probe answered.
- The kernel log captured during the restore has no NVRM, Xid or OOM line.
- Memwatch's low point since the restore is 11.06 GiB.

## Admission derived from this run (for the promotion commit)

- `memory_model.overhead_gib` = 105.74 − 77 − 18.0 = 10.74, rounded up to
  **10.8**. Candidate 9's value was 10.7.
- `safety.minimum_start_gib` = ceil(105.74 + 8) = **114**, unchanged.

## Not done here

**Promotion (plan A3) waits for the owner.** Per PREREG, the owner sees the
CJK diagnostic above next to run 2's ratios before promotion.
