# qwen38fn vLLM patch series (runtime-002)

These patches turn `native-runtime-001` into `native-runtime-002`. runtime-001 is vLLM nightly `72d59adc2` (prebuilt aarch64 cu130 wheel) plus #58439.

Each patch is Python or Triton only, so no CUDA build is needed. They apply in order with `git apply`, run in `runtime/lib/python3.12/site-packages`.

**Where things are pinned:** `configs/build-manifests/qwen38-flash-next-runtime-002.json` pins every file's sha256, the upstream head it came from, and the patched tree's digest. `scripts/tests/test_qwen38fn_tune_contract.py` checks them.

| Patch | Source | What it does | Switch |
|---|---|---|---|
| 0001 | vLLM #58439 @ `fa537316` (open) | Checkpoint-mapped PLE table: rows are read in place from the safetensors on NVMe. This is runtime-001's applied diff, byte for byte | `--engram-config {"checkpoint_mapped":true}` |
| 0002 | vLLM #59528 (merged as `240785b8` after this nightly) | Keeps padding and dummy rows of the QSA ring out of the shared null block | always on (fix) |
| 0003 | vLLM #58449 (open) | Fused draft-metadata updates for the QSA side cache. MTP stops logging "Fused multi-step draft decode is not supported" | always on |
| 0004 | vLLM #58737 (open) | Fuses the mamba align block-table gather into one Triton kernel. The output is bit-identical | always on |
| 0005 | jschmied/vllm #4 @ `ae0f7fdf` (open, on #58439) | `madvise(MADV_WILLNEED)` on a decode step's PLE rows before faulting them in, and starts the lookup at its own layer | via 0006 |
| 0006 | local | Puts 0005 behind `QWEN38FN_PLE_WILLNEED`. It is read once at import, defaults to off, and logs one startup line | `QWEN38FN_PLE_WILLNEED=1` |
| 0007 | salient-data/qwen38-next-dgx-recipe patch 0002 @ `0b46ed90` (Apache-2.0), which ports tonyd2wild's `mtp_draft_vocab.py` (Apache-2.0) | Reduced-vocabulary MTP drafting. **Only the `qwen4_exp/nvidia/mtp.py` part is included**: the FP8 dense-cast hunks are left out because the owner decided on lossless levers only (2026-10-11) | via 0009 |
| 0008 | vLLM #59973, as ported by the same recipe's patch 0003 (open) | NVFP4 copy of the drafter's lm_head; with a reduced vocabulary, only the draft rows are copied | `--speculative-config` `draft_lm_head_quantization: "nvfp4"` |
| 0009 | local | Reads `QWEN4EXP_DRAFT_VOCAB` once when the drafter is built, validates it, and rebinds `compute_logits`, so no draft step reads the environment. Removes the recipe's FP8 lm_head rebuild | `QWEN4EXP_DRAFT_VOCAB=65536` |

**Every lever is on the draft side or is I/O only.** The target model still runs the same weights, KV dtype and verification. So greedy output should match runtime-001's; the screen records whether it does.
