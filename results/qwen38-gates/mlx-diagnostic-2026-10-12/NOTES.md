# Qwen3.8-27B on MLX vs llama.cpp Metal (diagnostic, one run)

Diagnostic only, not qualification evidence. One speed run on 2026-10-12,
apple-silicon M5 Max 128 GiB, macOS 26.6. The Unreal editor in another
session was open but idle (no renders), as during the llama.cpp b11514 run
it is compared with (`../apple-silicon-128g-b11514-2026-10-11/`).

## Stack

- mlx-vlm 0.7.6, mlx 0.32.3 (Python 3.12 venv; `mlx_vlm.server`)
- target `mlx-community/Qwen3.8-27B-8bit` @ `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9`
  (affine 8-bit, group 64; shard sha256s verified against the HF LFS digests)
- draft `mlx-community/Qwen3.8-27B-MTP-8bit` @ `e88e48d055732ad75d9435f3059139d5279f2064`
  (`--draft-kind mtp`; the server logged "speculative decoding enabled";
  it reports no acceptance statistics)
- `mlx_vlm.server --host 127.0.0.1 --port 8016 --model <target> --draft-model <draft> --draft-kind mtp`,
  armed under `scripts/06_memwatch_macos.sh` (10 GiB floor)

Speed: `scripts/30_bench_speed.py` with the qualification kit's speed-cell
argv (reps 2, warmup 1, max_tokens 384, seed 42, ctx 0 and 28672), output
tokens counted with the GGUF tokenizer in both runs. MLX ignores the
llama.cpp `ignore_eos`/`min_tokens` extensions; every rep still produced
>= 200 tokens (all cells valid).

## Result

| | llama.cpp b11514 Q8_0 + MTP | MLX 8-bit + MTP |
|---|---|---|
| decode tok/s, ctx 0 | 31.44 | 23.84 |
| decode tok/s, ctx 28672 | 14.56 | 11.03 |
| TTFT s, ctx 28672 | 136.5 | 128.2 |

llama.cpp decodes about 32% faster at both contexts; MLX's 28K prefill is
about 6% faster. MLX is not a clear win for this model on this host, so no
MLX engine was added. Not measured: tool calls, vision, accuracy, memory
over a long session.
