# Run GLM-5.3-Flash on this Spark

GLM-5.3-Flash is an optional engine. Qwen (`qwen38-1m`) stays the recorded and
reboot default. Only one large model runs at a time.

One profile exists: `glm-5.3-flash/cuda-spark-128g-1m` (alias `glm53-1m`).
It serves 1,048,576 tokens as four native 262,144-token slots with tools,
reasoning, up to four 512x512 images and one 16-frame video per request.

## Dev serving (no sudo)

```bash
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m --port 8015 start
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m status
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m stop
```

Client settings: base URL `http://127.0.0.1:8015/v1`, model `glm-5.3-flash`.
Startup needs at least 110 GiB available memory and the inference lock; the
launcher verifies the pinned runtime and weight digests before it starts, and
the memory watchdog kills the server if available memory drops under 10 GiB.
Weights load in about 20 s (instanttensor); the server reports ready in under
two minutes.

## Production switch (owner-delegated sudo, already installed)

```bash
sudo -n scripts/52_engine_switch.sh glm53-1m     # switch in
sudo -n scripts/52_engine_switch.sh qwen38-1m    # switch back to the default
scripts/52_engine_switch.sh status --json
```

The switch is available only once the profile status is `qualified`
(see [STATUS](../results/glm53-flash-gates/STATUS.md)).

## Prove speed and fidelity

```bash
scripts/94_qualify_profile.py --profile glm-5.3-flash/cuda-spark-128g-1m --with-soak
```

See [QUALIFY-PROFILE](QUALIFY-PROFILE.md). Every number in the README row and
in [STATUS](../results/glm53-flash-gates/STATUS.md) comes from one such run,
`results/glm53-flash-gates/qualify-2026-09-11-rerun/` (about 80 minutes with the
soak and the four-slot 1M fill). Measured there, against the `qwen38-1m`
targets declared in the profile's `qualification_targets`:

| cell | GLM-5.3-Flash | qwen38-1m target | status |
|---|---|---|---|
| decode tok/s @ 0 ctx | 20.12 | >= 17.46 | PASS |
| decode tok/s @ 28,672 | 19.84 | >= 26.71 | FAIL (Qwen uses MTP; GLM's MTP layer cannot run on this stack) |
| prefill tok/s @ 28,672, cold prefix | 513.3 (warm prefix cache ~3,400) | >= 698.7 | FAIL |
| TTFT short prompt / @ 28K | 1.75 s (reps 1.20 and 2.29) / 58.93 s | <= 0.5 s | FAIL |
| tool-call probe | 20/20 | >= 19 | PASS |
| MMMU-val-100 (thinking mode; chat mode 20%, unparseable answers) | 77% | >= 64% | PASS |
| media at declared maximum (4 x 512px, 16-frame 512px video, 5th image -> 400) | PASS | PASS | PASS |
| dNLL vs BF16 teacher logits (25 x 2,047 positions) | 0.079 (upper-95 0.106); top-1 loss 1.49 pp (upper-95 1.91); agreement 87.3% | <= 0.01 | FAIL (unreachable at 2 bpw) |
| four-slot 1M fill | 1,000,560 tokens, 4/4 needles, low point 13.30 GiB | PASS, floor 10 GiB | PASS |
| 30-minute soak | 136 requests, 0 errors, 20.56 tok/s median, low point 14.15 GiB | | PASS |

## What is under the hood

- Runtime `native-runtime-003` (`~/.cache/glm53-flash/native-runtime-003/runtime`):
  vLLM fork ZJY0516/vllm `878631b6` + `patches/glm53/vllm.patch` (sm_121 sparse
  MLA), `vllm-exl3` 0.4.2 (`d3cfd394` + `patches/glm53/vllm-exl3.patch`),
  exllamav3 1.4.9 (+ `patches/exllamav3-v1.4.9-aarch64.patch`), flashinfer
  0.6.18rc10, torch 2.13.0+cu130. Build record: `native-runtime-003/REPORT.md`
  and `manifest.json` next to it. Flashinfer JIT-compiles two kernels at first
  use, so the profile's `HOME` carries a seeded JIT cache and a `ninja` binary.
- Weights: `exl3-k2-densek4-mtp` in `configs/profiles/glm-5.3-flash/model.json`
  (pack A: vcruz305 K2 2-bit routed experts + turboderp 3/4-bit dense overlay,
  plus the K2 MTP layer-45 shard). Pack B (`exl3-turboderp-2.05`) is pinned and
  hashed but does not load in this fork (`KeyError o_proj.suh`, follow-up).
- KV cache: fp8 DeepSeek sparse MLA, `--kv-cache-memory-bytes 10600000000`
  = exactly 1,048,576 tokens with prefix caching on (the hybrid model's prefix
  cache costs ~10% of KV tokens; agent loops that resend a prefix prefill at
  ~3,400 tok/s instead of ~500).
- Speed settings: FULL_DECODE_ONLY CUDA graphs (no `--enforce-eager`),
  2,048-token prefill batches, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (without it the caching allocator fragments under four concurrent 250K
  prefills and MemAvailable drops 5-7 GiB; see
  `results/glm53-flash-gates/context-2026-09-11-diagnosis/`).
- MTP speculative decoding is NOT enabled: the SM120 sparse-MLA decode kernel
  has no shape for the MTP layer's kpool-widened top-k (2,176), so vLLM fails
  at graph capture (`results/glm53-flash-gates/speed-2026-09-10/mtp-attempts/`).
- Two ASGI middlewares (`scripts/lib/glm53_runtime_policy.py`): media admission
  (4 images / 1 video, 16 MiB body) and an SSE shim that splits vLLM's
  finish_reason-on-last-delta into the llama.cpp shape every harness here
  expects. A worker subclass (`scripts/lib/glm53_worker.py`) releases the
  load-time allocator cache so idle MemAvailable is ~16 GiB instead of ~10.
- Background: `GLM53-INTEGRATION-RESEARCH.md`. Everything the previous branch
  produced (590 commits, ~300 experiment directories) is preserved on
  `archive/glm-5.3-flash-cuda-astra-20260910`.
