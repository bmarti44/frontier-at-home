# Track B gate 1 — SGLang arms on GB10 — 2026-08-20

Pinned image lmsysorg/sglang@sha256:febfb971… (docker), loopback :30000,
--mem-fraction-static 0.50, --mamba-ssm-dtype float32, docker caps
--memory/-swap 100g, DSV4 stopped per window and restored.

## Strict cells (30_bench, suite-valid) + greedy code probes

| arm | decode @0 / @28K | prefill @28K | TTFT @28K / short | code probe |
|---|---|---|---|---|
| fp8 (no spec) | 7.85 / 7.29 | 819 | 42.4 s / 0.17 s | 7.9 |
| nvfp4 (no spec) | 12.33 / 11.63 | **1724** | **20.2 s / 0.12 s** | 12.3 |
| nvfp4 + DSpark | — (see below) | — | — | **30.8 / 30.7** (two runs) |

- SGLang prefill is the standout: 1724 tok/s @28K (2.5x the qualified
  llama.cpp profile's 699), TTFT 20 s vs 50 s at depth.
- DSpark lifts nvfp4 code decode 12.3 -> ~30.7 (2.5x), matching the
  community's 34-40-on-code band (our probe is mixed code).
- Strict 30_bench cells are NOT computable with DSpark on: the draft
  emits ~3-7-token verified blocks per stream event, so per-token
  timestamping undercounts ("early stop: 85 timestamped tokens") —
  a harness/streaming granularity mismatch, same class as the GLM
  THINKING accounting caveat. Spec speed is recorded via wall-clock
  probes (usage.completion_tokens / elapsed).

## Negative results / incidents (preserved)
- DFlash2 draft (z-lab, community fork's variant) is NOT supported by
  this image (`DFlash2DraftModel` unregistered) — arms use the natively
  supported DSpark draft (auto_map + models/dspark.py).
- fp8 + DSpark does not fit at mem-fraction 0.50 with the template's
  --max-mamba-cache-size 96 (sized for the 23 GB NVFP4 checkpoint;
  FP8 is 31 GB): KV pool exhausted at startup. Deferred: retry with a
  smaller mamba cache if the FP8+spec arm is ever needed.
- Serve wrapper fixes en route: /run/dsv4 state perms, --rm log loss,
  stale-container handling, SGLang ignore_eos vs llama.cpp extension.

Gate 2 (DSpark greedy-equivalence + accuracy suites on nvfp4-spec,
ledger namespace trackb-nvfp4) follows.
