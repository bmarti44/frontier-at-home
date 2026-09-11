# GLM-5.3-Flash gate status

Updated 2026-09-11 (branch rebuilt from main; previous work archived on
`archive/glm-5.3-flash-cuda-astra-20260910`). Profile:
`glm-5.3-flash/cuda-spark-128g-1m` (alias `glm53-1m`), four 262,144-token
slots = 1,048,576 tokens, tools, images, 16-frame video. Optional engine;
`qwen38-1m` stays the default.

Evidence bundle: `qualify-2026-09-11/` (one `94_qualify_profile.py` run:
speed, tool-call, vision, media, teacher logits, 30-minute soak, four-slot 1M
context). `qualify-2026-09-11-cells/` is the preceding dev run on the same
server (1,024-token prefill batches) and `context-2026-09-11/` the standalone
1M gate; numbers below are from `qualify-2026-09-11/SUMMARY.md`.

| Gate | Result | Target (`qwen38-1m`) | Evidence |
|---|---|---|---|
| Runtime builds from pinned sources (sm_121) | PASS | | `build-*`, `~/.cache/glm53-flash/native-runtime-003/REPORT.md` |
| Weights pinned and hashed | PASS | | `configs/profiles/glm-5.3-flash/model.json`, `weights/glm-5.3-flash/manifest.json` |
| Decode tok/s @ 0 ctx | **PASS** 20.06 | >= 17.46 | `qualify-2026-09-11/speed/` |
| Decode tok/s @ 28,672 | **FAIL** 19.91 | >= 26.71 (Qwen MTP) | same |
| Prefill tok/s @ 28,672 (cold prefix) | **FAIL** 512.5 | >= 698.7 | same; warm prefix-cache prefill ~3,400 tok/s (`speed-2026-09-10/candidate-graphs.json`) |
| TTFT short prompt | **FAIL** 5.06 s | <= 0.5 s | same |
| Tool-call probe | **PASS** 20/20 | >= 19/20 | `qualify-2026-09-11/toolcall/` |
| MMMU-val-100 vision | **PASS** 73 | >= 64% | `qualify-2026-09-11/vision/` |
| Media at declared maximum (4 x 512px images, one 16-frame 512px video, 5th image rejected) | **PASS** | | `qualify-2026-09-11/media/` |
| Fidelity vs BF16 teacher logits (25 windows x 2,047 positions) | **FAIL vs 0.01 gate**: dNLL 0.079 (upper-95 0.106), top-1 loss 1.53 pp, top-1 agreement 87.2% | repo gate 0.01 dNLL | `qualify-2026-09-11/teacher/` |
| GSM8K / MMLU-Pro / HumanEval vs Qwen | SKIPPED: no GLM chat encoder in `31_bench_accuracy.py` | | |
| Four 262,144-token slots, direct 1M fill, floor 10 GiB | **PASS**: 1,000,560 tokens, 4/4 needles + controls, low point 13.38 GiB | 1,048,576 cap | `qualify-2026-09-11/context/`, `context-2026-09-11/`, diagnosis of the earlier breaches in `context-2026-09-11-diagnosis/` |
| 30-minute soak | **PASS**: 136 requests, 0 errors, median 20.63 tok/s, low point 14.23 GiB | | `qualify-2026-09-11/soak/` |
| Production switch + rollback (`glm53-1m` -> `qwen38-1m`) | **PASS** 2026-09-11: in 54 s, back 54 s, default restored, auth proxy 401 unchanged | | `switch-2026-09-11/` |

## Unmet requirements (honest list)

- **Decode at 28K (~20 tok/s) and prefill (~500 tok/s cold) are below Qwen's
  26.7 / 698.7.** Qwen's 28K decode number comes from its MTP draft; GLM's
  MTP layer cannot run on this stack (SM120 sparse-MLA decode kernel has no
  shape for the MTP layer's kpool-widened top-k 2,176; see
  `speed-2026-09-10/mtp-attempts/`). Larger prefill batches (4,096) gave no
  gain. Decode at 0 ctx (20.0) beats Qwen (17.5).
- **Short-prompt TTFT ~1.7-5 s vs 0.4 s.** vLLM chunked-prefill + reasoning
  parser startup per request; not tuned further.
- **0.01 dNLL gate**: unreachable for any 2-bit pack of a 320B model in 120 GiB
  (FP8 alone is ~0.03). Measured dNLL is reported for the owner's decision;
  reasoning/termination windows are at teacher parity, legal prose is worst.
- **Accuracy suites**: no GLM chat encoder is registered for
  `31_bench_accuracy.py`; the cell is SKIPPED, not passed.
- **Pack B (turboderp 2.05 bpw)** is pinned but does not load in the fork.

## Kit check against the default

`results/qwen38-gates/qualify-2026-09-11-kit-check/` is the same kit run on
`qwen38-1m` (speed, tool-call, vision cells): decode 22.08 tok/s @ 0 / 28.63 @
28,672, prefill 690.9 tok/s @ 28K, TTFT 0.51 s short prompt, tool-call 19/20,
MMMU-val-100 64% — the README row within noise, so the GLM numbers above are
comparable measurements, not a different harness.
