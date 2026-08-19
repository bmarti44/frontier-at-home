# Qwen3.8-27B smoke gate — 2026-08-18

Engine: llama.cpp b10488 (9d77fa17), CUDA sm_121, server sha bcef273b…
(configs/build-manifests/llamacpp-qwen38-9d77fa17.json).
Model: Qwen3.8-27B-Q4_K_M.gguf (sha e103abf9…, matches bartowski HF LFS oid),
mmproj-Qwen3.8-27B-f16.gguf, port 8015 loopback, `-ngl 99 -fa on --no-mmap
-c 32768 --parallel 1`. Served inside a transient systemd user unit
(MemoryHigh=45G / MemoryMax=50G / MemorySwapMax=0 / OOMPolicy=kill); DSV4
production stopped for the window and restored after (verified healthy on
127.0.0.1:8013).

## Results — ALL PASS

1. **Gated-DeltaNet canary** (raw `/completion`, greedy, 64 tokens):
   coherent continuation of "The three primary colors are" (`canary.txt`).
   The pre-b10450 CUDA corruption bug is absent in this build.
2. **Text sanity** (chat, greedy): 17*23 → "391"; capital of Australia →
   "Canberra". (p3 haiku returned empty content at max_tokens=200 —
   thinking-segment consumption; both MTP arms identical so equivalence
   still binds. Gates use larger budgets / explicit thinking mode.)
3. **Vision sanity**: 3 pinned MMMU cases end-to-end (base64 image + MC
   question), 2/3 correct, 0 transport errors, 0 unparseable
   (`vision-sanity/`). Vision path (mtmd + mmproj) works.
4. **MTP greedy equivalence**: `--spec-type draft-mtp --spec-draft-n-max 2`
   vs off, 3 prompts, greedy — byte-identical on all three
   (`mtp0-p*.txt` vs `mtp1-p*.txt`). MTP is exact under greedy; safe to
   use for speed.

## Incidents (no fidelity impact)

- First attempt bound port 8014, which is the dsv4-authhelper listener —
  Qwen moved to **8015**.
- llama-cli canary entered an interactive REPL despite `-no-cnv` and was
  replaced with a server-side raw `/completion` canary.
- Restore-health checks initially hit the auth-fronted 8010 (401) and
  reported false restore failures; production was verified healthy on 8013
  both times. Gate scripts now check 8013.
