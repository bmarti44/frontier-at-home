# Qwen3.8-27B speed gate — 2026-08-18

Engine: llama.cpp b10488 (9d77fa17), CUDA sm_121; port 8015 loopback;
`-ngl 99 -fa on --no-mmap --parallel 1`, mmproj loaded; transient capped
systemd units; DSV4 stopped per window and restored (verified on 8013).
Bench: scripts/30_bench_speed.py, reps 2 + 1 warmup, Qwen tokenizer
(`--output-tokenizer-path` + sha 0997f410…), `--extra-body
{"chat_template_kwargs":{"enable_thinking":false}}` (fixture is completion-
style; thinking off for clean token accounting). All cells strict-valid.

Shallow cells served with `-c 32768`; 28K cells re-run with `-c 40960`
because the 28672-fixture renders to 34,673 Qwen tokens (first attempt's
deep cells 400-errored on context; preserved in the shallow-*.json files).

## Decode (median tok/s)

| Config | ctx 0 | ctx 28672 |
|---|---|---|
| Q4_K_M raw            | 11.59 | 10.21 |
| **Q4_K_M + MTP (n=2)** | **19.34** | **18.36** |
| Q6_K raw              |  8.84 |  8.05 |
| Q6_K + MTP (n=2)      | 16.42 | 16.23 |

Prefill @28K (incl. queue+setup): Q4_K_M ~719 tok/s, Q6_K ~642 tok/s.
TTFT @28K ~48 s (Q4).

## Reading

- Raw Q4 matches the community llama.cpp figure for this model on GB10
  (11.6) exactly; decode is bandwidth-bound (Q6/Q4 ratio tracks file size).
- MTP (`--spec-type draft-mtp --spec-draft-n-max 2`) is greedy-exact
  (smoke gate) and lifts Q4 to 18.4 tok/s at 28K — DSV4-parity decode
  (18) with ~17x DSV4's prefill. Q6_K+MTP lands 16.2 at depth.
- Hybrid DeltaNet attention barely droops with depth (11.6→10.2 raw).
- MTP acceptance is content-dependent (community: code >> prose); this
  fixture is prose-like, so these are conservative MTP numbers.
- Untested free headroom: `--spec-draft-n-max 3` + `--spec-draft-p-min
  0.6-0.75` (greedy-exact tune, community-reported gains on
  bandwidth-limited hardware).
