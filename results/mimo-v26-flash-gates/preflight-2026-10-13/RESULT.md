# MiMo-V2.6-Flash M3 preflight: result (DIAGNOSTIC)

**Outcome: every path ran and works on TabbyAPI, but peak use was 106.09 GiB,
0.09 GiB over the 106 GiB limit.** Under the PREREG's fixed rule, M3 does not
start at this configuration. I stopped and reported to the owner.

These numbers are diagnostic only. They are never headline, README or
acceptance evidence.

## Window

- **Head:** `fca575d3`, clean.
- **Production:** `qwen38fn` stopped at 19:14:37Z and restored at 19:47:01Z
  (rc 0, first attempt), about 32 min.
- **Load:** launched 19:14:50Z, ready after 55 s. MemFree's low during the
  load was 8.31 GiB (r3: 6.38).
- **Safety:**
  - 0 kill lines in the kernel, server and memwatch logs;
  - swap growth 0.000 GiB;
  - 1,322 one-second samples, at most 2 s apart;
  - scope empty after stop.

## Memory (the rule's clause)

| | value |
|---|---|
| MemAvailable at launch | 115.35 GiB |
| MemAvailable low | **9.25 GiB**, at 19:19:46Z, the start of the 250K fill |
| peak use (launch − low) | **106.09 GiB** (limit 106) |
| `ceil(peak + 8)` | 115 (the switch admission is still ≤ 115) |
| scope `anon` at ready → at the low → maximum | 3.09 → 4.56 → 4.56 GB |
| host anon outside the scope at the low | about 0.7 GiB (the kit's clients and the system) |

- **Growth in the scope's anon.** It grew by about 1.47 GB over the run. That
  fits the new caps: up to 1 GiB of sliding-window checkpoints (49.25 MiB
  each) plus the image cache (≤ 256 MB).
- **Comparison with r3.** r3 reached 105.59 GiB without the context fill and
  without vision. The 262K fill itself added little: MemAvailable held at
  9.7–10.0 GiB through its 684 s.

## Paths

| path | result | compatible |
|---|---|---|
| probe | greedy hashes byte-identical to r3 on all three prompts | yes |
| agent gate (`--engine vllm --prefix-block-tokens 256`) | **4/4** (see below) | yes |
| context, direct fill | **PASS**: server prompt 250,135 tokens (client 250,128), needles correct, controls OK, TTFT 684.3 s, 79 output tokens, `stop` | yes |
| vision (MMMU-val first 20, chat) | 11/20; **4 cut at the 256-token budget** (`finish_reason length`); 11/16 of answers that finished | **harness mismatch, closed** |
| accuracy encoder path (GSM8K dev) | **98/100**, 0 invalid, 0 fallback, finished before the 1,200 s cut | yes |

**Agent gate:**
- prefix-cache: an exact repeat reused 22,272 of 22,511 tokens (within one page);
- turn-continuation: turn 2 re-ran 39 tokens;
- slot-thrash: turn 2 re-ran 355 tokens after the utility call;
- prefill-throughput: a novel 22.5K-token prompt at 812 tok/s (27.8 s).

**Vision mismatch, closed under the rule.** It was fixed in the shared kit,
RED-first (`1ef0be30`, `ff4d6c0a`). `qualification_options.vision_max_tokens`
reaches `38_bench_vision.py` as `--max-tokens`. MiMo declares 4,096; every
other profile keeps 256. The fix is client-side, so it needs no preflight
rerun.

The context cell's 512-token budget was enough: MiMo answered in 71 and 79
tokens. So the fallback fill was not needed and did not run.

## What the owner decides

The rule leaves the next step to the owner. The options and their trade-offs
are in the claim PR and in this session's report. The evidence above is
unchanged whatever is chosen.
