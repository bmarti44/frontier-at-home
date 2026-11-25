# Candidate 2 result: PASS

The frozen scorer (`gate_verdict.py`, freeze `4bb1f7e0`, sha256 prefix `f55fd5315cdbe888`) returns **PASS** in `gate-summary.json`, with no failed clause, no missing evidence and no `GATE_*` override.
- Before the freeze, `gate_mutations.py` matched **75/75**: the 73 synthetic cases plus the two real-evidence checks (screen carrier a6 and Flash-Next c10).
- The run ran at head `4bb1f7e0`, the freeze itself; the kit's manifest head is the same. The FROZEN diff against the freeze was empty.
- No public randomness was drawn (one arm, fixed fixtures; PREREG "Freeze and randomness").
- Window: launch 08:30:16Z, ready after 192 s, kit cells 08:34:24Z to 09:35:30Z, scope empty 09:35:34Z. Production `qwen38fn` was restored at 09:47:08Z (`restore rc=0`), with an empty kernel log for the run and the restore.

| Clause | Result |
|---|---|
| 1. Kill lines (kernel, server, memwatch) | none |
| 2. Swap growth ≤ 0.25 GiB | 0.0 GiB |
| 3. Low point ≥ 10 GiB | 24.35 GiB (during the 1M fill); 4,615 samples |
| 4. Admission `ceil(peak + 10)` ≤ 115 | peak 91.22 GiB (launch 115.58 − low 24.35), so 102 |
| 5. Direct 1M | KV capacity 1,238,000 tokens at 262,144 per request; fill PASS, 4 × 250,128 = 1,000,572 tokens, needles 4/4, controls OK; cap 1,048,576 |
| 6. Text path vs screen carrier a6 | 3/3 greedy hashes identical (`170c63d110`, `85f745a0d8`, `217aaabaf8`) |
| 7. Tool-call ≥ 19/20 | 20/20 |
| 8. Agent gate | 4/4 (vLLM, prefix block 64, DSpark block drop) |
| 9. Soak at the 5 GiB default floor | PASS: 257 requests, 0 errors, minimum 24.65 GiB, decode median 37.38 tok/s, 1,801 s |
| 10. Vision (MMMU-val-100) | 77/100, 4 invalid, 0 errors; z = +2.02 against the default's 64 |
| 11. Speed suite valid at K = 8, block-aware method | valid at 0 and 28,672 |
| 12. Host page-cache release worker | runtime releases under `~/.cache/inkling-2-mini` (105,325 files) and `~/.cache/qwen38-flash-next` (136,225); one model-root release under `~/models` (919 files) with MemFree 88.39 GiB at the 63.33 GiB arena + 2 GiB |

Artifact identity at the start (`verify_identity.py`): interpreter match; vLLM tree `5bcb020bc1e801e3` (6,525 files, after patches 0001–0004) match; package freeze `85a4f99d5e707f57` match; weights 14 files, 20,191,643,333 bytes, 0 mismatches. Every lever line logged: DSpark, humming MoE, `INKLING2_MAX_CTAS_POW2=1`, batch 2,048, effort low.

## This profile's speed (clause 11; the M5 promotion rule reads these)

The kit's speed cell is the same frozen bench as Flash-Next's c10 (`30_bench_speed_blocks.py`, bench sha256 `247c60f5…`, fixture `ctx-32k.txt`, seed 42, 2 reps, 384 client tokens per rep, temperature 0). Only K differs: 8 here (DSpark ×7), 4 for Flash-Next (MTP ×3).

| Metric | Inkling 2 Mini, c2 | Flash-Next c10 (default) |
|---|---|---|
| Prefill at 28,672 | **3,332.5 tok/s** | 2,711.5 |
| Decode at 0 | **36.14 tok/s** (reps 33.48, 38.80) | 34.07 |
| Decode at 28,672 | **44.16 tok/s** (reps 39.21, 49.12) | 40.92 |
| TTFT at 0 / 28,672 | 0.17 s / 9.72 s | 0.21 s / 12.80 s |

Decode with DSpark depends on the text. The two reps at 28,672 differ by 25%, and the screen's three reps of the same configuration gave 49.05, 37.53 and 38.25 (median 38.25; `../tune-screen-2026-11-23/a6`, diagnostic). The kit's two-rep median is the preregistered method for both models, so it is the number the rule compares; the spread is reported here so the margin is read with it.

## Reported, not gating

- **DSpark acceptance length** (server metrics, 334 intervals over the whole run): median 2.21, range 1.82–7.09.
- **1M fill:** largest TTFT 551.1 s; total wall 568.8 s.
- **Load:** ready 192 s after launch; idle MemAvailable 26.8 GiB.
- **Admission for M4:** `minimum_start_gib = ceil(91.22 + 10) = 102`.

## Review notes carried from the freeze review (lows, no blocking finding)

- `run-c2.sh` line 10's "Adapted from" comment names `mimo-v26-flash-gates/qualify-2026-10-13-c1/run-c2.sh`; the source is that directory's `run-c1.sh`. The candidate's file rename also rewrote this comment. It is a comment only.
- PREREG.md says "73 cases"; the suite reports 75, the two extra being the real-evidence checks the same paragraph describes.

Candidate count for the M3 gate: 2 (candidate 1 superseded before it ran).
