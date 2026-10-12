# llama.cpp pin choice (T2.1, orchestrator-performed)

**Pinned commit: `32e789fdfd598e9a1872da55ac941e4d94f030bd`** (upstream `ggml-org/llama.cpp` master, 2026-07-16).

## Why this commit

DeepSeek-V4 support merged upstream on 2026-06-29 (`8c146a836`, PR #24162). The pinned commit is the newest master at clone time and contains every V4-relevant follow-up present in upstream history:

| Commit | What it fixes/adds | Why it matters here |
|---|---|---|
| `8c146a836` | DeepSeek V4 architecture (PR #24162) | the model itself |
| `00f5442cc` | `GGML_OP_LIGHTNING_INDEXER` (PR #24231) | V4 sparse-attention indexer op |
| `024c46ae4` | fix quantized KV cache for dsv4 (PR #25202) | the "quantized-K garbage output" bug from the plan review — fixed upstream; we still run fp16 K cache at baseline out of caution |
| `2ed3c1abb` | f16 KQ masks w/ FA, remove raw_k repeats in V4 (PR #25370) | correctness/perf with flash attention |
| `33a75f41c` | DeepseekV4: reduce graph splits (PR #25702) | decode performance |
| `d1b34251b` | spec: add DFlash speculative decoding (PR #22105) | potential dev-phase speed uplift (needs a DFlash-format draft model — availability unverified) |
| `2969d6d15` | MTP speculative decoding infra (`draft-mtp`) | potential dev-phase uplift if the Unsloth GGUF retains the MTP head (unverified) |

## Baseline mitigations retained (from plan review round 1)
Single slot (`-np 1`), RAM prompt cache disabled, fp16 K cache, explicit `-c`/batch sizes. Speculative decoding stays OFF at baseline; `draft-mtp`/DFlash are dev-tuning experiments only, measured on the dev split.

## Risk note
Master-of-today can carry non-V4 regressions; accepted because (a) the SHA is frozen (no drift), (b) golden correctness tests gate before any benchmark, (c) the alternative (the bare merge commit) lacks the three correctness/perf fixes above.

## Portable pin: b11514

**Pinned commit: `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`** (tag `b11514`,
2026-10-08), registered as `portable` in `configs/pins/llamacpp.json`.

Every off-Spark profile on upstream llama.cpp uses it: Apple Silicon, Strix
(ROCm/Vulkan), consumer CUDA and CPU. Spark-qualified engines keep their own
pins (`9d77fa17`, `0dc74e33`, `06f8cebd`) until they requalify on the Spark.

It is the last tag before the 2026-10-09 flash-attention kernel additions
(#30209), and it still includes the Metal correctness fixes from 2026-10-06/07.

| PR | What it adds or fixes | Why it matters here |
|---|---|---|
| #27461 | Metal 4 tensor API on M5+/A19+ | M5 prefill. b10488 predates it |
| #29570 | tensor-API flash attention for F16 KV | M5 long-context prefill |
| #29869 / #30065 | few-row MMA mat-mul for speculative decoding | MTP/DFlash verify batches on Metal |
| #28068 | GDN q/k normalization `rsqrt` (matches FLA/vLLM) | Qwen3.8 fidelity on every backend. KLD 0.00177 → 0.00175 on 27B Q4_K_M |
| #25585 / #29169 | DSV4 fused hyper-connection ops; Metal hc support | off-Spark DSV4 no longer "pre-fusion" |
| #29340 / #30100 / #29220 | Metal FA threadgroup overflow, MUL_MAT+ADD fusion, FA mask bounds | correctness |

Known open issues at this pin (also listed in the registry):

- #28748: a Metal tensor-API tile more than 2 GiB from a tensor's start
  miscomputes. This hits very large tensors only (for example the Flash-Next
  embedding table), not Qwen3.8-27B.
- #30174: Metal `draft-mtp` grows host memory by up to ~15 GB over a long
  session.
- #30175: flash-attention prefill is nondeterministic on RDNA3 ROCm with
  multimodal requests.

Not moved: the Spark's qualified Qwen3.8-27B engine (b10488) lacks #28068.
Adopting it there is a Spark requalification (AGENTS.md acceptance gates),
not a config change.
