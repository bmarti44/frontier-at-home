# GLM-5.3-Flash gate status

Updated 2026-09-11 (branch rebuilt from main; previous work archived on
`archive/glm-5.3-flash-cuda-astra-20260910`). Profile:
`glm-5.3-flash/cuda-spark-128g-1m` (alias `glm53-1m`), four 262,144-token
slots = 1,048,576 tokens, tools, images, 16-frame video. Optional engine;
`qwen38-1m` stays the default.

Evidence bundle: `qualify-2026-09-11-rerun/` (one `94_qualify_profile.py` run
of every cell on 2026-09-11 14:37Z, then the vision cell re-measured in the
model's thinking mode; the chat-mode transcripts stay in `vision-chat-mode/`).
`qualify-2026-09-11/` is the earlier run whose SUMMARY.md carried a vision
figure from a different cell run (see its `SUPERSEDED.md`); the kit bug that
left the stale file is fixed. `qualify-2026-09-11-cells/` is the preceding
dev run and `context-2026-09-11/` the standalone 1M gate; numbers below are
from `qualify-2026-09-11-rerun/SUMMARY.md`.

| Gate | Result | Target (`qwen38-1m`) | Evidence |
|---|---|---|---|
| Runtime builds from pinned sources (sm_121) | PASS | | `build-*`, `~/.cache/glm53-flash/native-runtime-003/REPORT.md` |
| Weights pinned and hashed | PASS | | `configs/profiles/glm-5.3-flash/model.json`, `weights/glm-5.3-flash/manifest.json` |
| Decode tok/s @ 0 ctx | **PASS** 20.12 | >= 17.46 | `qualify-2026-09-11-rerun/speed/` |
| Decode tok/s @ 28,672 | **FAIL** 19.84 | >= 26.71 (Qwen MTP) | same |
| Prefill tok/s @ 28,672 (cold prefix) | **FAIL** 513.3 | >= 698.7 | same; warm prefix-cache prefill ~3,400 tok/s (`speed-2026-09-10/candidate-graphs.json`) |
| TTFT short prompt | **FAIL** 1.75 s (reps 1.20 / 2.29) | <= 0.5 s | same |
| Tool-call probe | **PASS** 20/20 | >= 19/20 | `qualify-2026-09-11-rerun/toolcall/` |
| MMMU-val-100 vision | **PASS** 77% in the model's thinking mode (9 unparseable); chat mode 20% (77 unparseable `X.X` answers) | >= 64% | `qualify-2026-09-11-rerun/vision/`, `vision-chat-mode/` |
| Media at declared maximum (4 x 512px images, one 16-frame 512px video, 5th image rejected) | **PASS** | | `qualify-2026-09-11-rerun/media/` |
| Fidelity vs BF16 teacher logits (25 windows x 2,047 positions) | **FAIL vs 0.01 gate**: dNLL 0.079 (upper-95 0.106), top-1 loss 1.49 pp, top-1 agreement 87.3% | repo gate 0.01 dNLL | `qualify-2026-09-11-rerun/teacher/` |
| GSM8K / MMLU-Pro / HumanEval vs Qwen | GSM8K **94/100** holdout (Qwen 98), MMLU-Pro **202/247** holdout (Qwen 210; 3 truncated at 16K); HumanEval **TBD** (stopped at 5/164: raw-prompt protocol makes GLM generate the full 16K budget per problem, ~37 h) | general target: Qwen row | `qualify-2026-09-11-rerun/accuracy/`, `HUMANEVAL-STOPPED.md`; encoder parity `encoder-parity-2026-09-11/` |
| Four 262,144-token slots, direct 1M fill, floor 10 GiB | **PASS**: 1,000,560 tokens, 4/4 needles + controls, low point 13.30 GiB | 1,048,576 cap | `qualify-2026-09-11-rerun/context/`, `context-2026-09-11/`, diagnosis of the earlier breaches in `context-2026-09-11-diagnosis/` |
| 30-minute soak | **PASS**: 136 requests, 0 errors, median 20.56 tok/s, low point 14.15 GiB | | `qualify-2026-09-11-rerun/soak/` |
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
- **HumanEval is TBD.** GSM8K (94%) and MMLU-Pro (81.8%) are measured
  through the `glm53` encoder; HumanEval's raw-completion protocol (no stop
  sequences, 16,384 tokens) makes GLM run to the budget on every problem
  (13.7 min each, ~37 h), and the owner stopped it at 5/164. A shorter
  budget would change what the extractor scores, so it was not substituted.
- **Pack B (turboderp 2.05 bpw)** is pinned but does not load in the fork.

## Kit check against the default

`results/qwen38-gates/qualify-2026-09-11-kit-check/` is the same kit run on
`qwen38-1m` (speed, tool-call, vision cells): decode 22.08 tok/s @ 0 / 28.63 @
28,672, prefill 690.9 tok/s @ 28K, TTFT 0.51 s short prompt, tool-call 19/20,
MMMU-val-100 64% — the README row within noise, so the GLM numbers above are
comparable measurements, not a different harness.
