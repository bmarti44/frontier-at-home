# MiMo-V2.6-Flash M3 preflight r2: result (DIAGNOSTIC)

**Outcome: every path ran, and peak use was 105.52 GiB, within the 106 GiB
limit.** The low point was 9.76 GiB. Under the PREREG's fixed rule, the M3 c1
PREREG is written with this configuration: sliding-window checkpoints capped at
512 MB, and vision at a 4,096-token answer budget.

These numbers are diagnostic only. They are never headline, README or
acceptance evidence.

## Window

- **Head:** `23467308`, clean.
- **Production:** `qwen38fn` stopped at 20:57:00Z and restored at 21:29:40Z
  (rc 0, first attempt), about 33 min. The driver exited rc 0.
- **Load:** launched 20:57:09Z, ready after 56 s. MemFree's low during the
  load was 8.80 GiB (first preflight: 8.31).
- **Safety:**
  - 0 kill lines in the kernel, server and memwatch logs;
  - swap growth 0.000 GiB;
  - 1,916 one-second samples, at most 2 s apart;
  - the scope was empty after stop.

## Memory (the rule's clause)

| | first preflight (1024 MB) | r2 (512 MB) |
|---|---|---|
| MemAvailable at launch | 115.35 GiB | 115.28 GiB |
| MemAvailable low | 9.25 GiB | **9.76 GiB**, at 21:02:46Z, at the end of vision |
| peak use (launch − low) | 106.09 GiB | **105.52 GiB** (limit 106) |
| `ceil(peak + 8)` | 115 | 114 |
| scope `anon`: ready → maximum | 3.09 → 4.56 GB | 3.09 → 4.11 GB |

Per phase, r2's lowest MemAvailable was:
- probe and agent gate: 10.85 GiB;
- vision: 9.76 GiB;
- the 250K fill: 9.88 GiB;
- GSM8K: 10.03 GiB.

The 512 MB cap took about 0.45 GB off the scope's anon peak, as expected.
After that, the r2 low sits in vision, where the image cache (≤ 256 MB) and
the longer 4,096-budget answers add to the checkpoints.

## Paths

| path | result |
|---|---|
| probe | greedy hashes byte-identical to r3 and the first preflight on all three prompts |
| agent gate (`--engine vllm --prefix-block-tokens 256`) | **4/4**: repeat reused 22,272 of 22,511; turn 2 re-ran 75 tokens; slot-thrash 355; novel prefill 811 tok/s (27.8 s) |
| context, direct fill | **PASS**: server prompt 250,135 tokens, needle correct, controls OK, TTFT 685.1 s, low 9.76 GiB (the cell's own sampling) |
| vision (MMMU-val first 20, chat, 4,096 budget) | 14/20, 0 invalid, 0 errors; all 20 `finish_reason stop`; the longest answer was 1,281 tokens |
| accuracy encoder path (GSM8K dev) | **98/100**, 0 invalid, finished before the 1,200 s cut |

The vision fix closed the first preflight's mismatch. No answer reached the
budget: ten answered in 2 tokens, and the other ten took 88–1,281 tokens.
Fourteen of twenty is a 20-case smoke result, not a vision measurement. The
gate measures vision on all 100 cases in M3.

## Next

Freeze M3 c1 (`../qualify-2026-10-13-c1/PREREG.md`) at this configuration. The
gate's peak clause uses the same formula and limit as this rule.
