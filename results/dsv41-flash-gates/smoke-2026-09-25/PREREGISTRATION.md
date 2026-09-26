# DeepSeek V4.1 Flash CUDA smoke: preregistration (2026-09-25)

**Kind:** fidelity/correctness smoke. It is **not** a context-capability or speed result.

## Frozen inputs
- Engine: antirez/ds4 `0aaea5a238fb41a35106a551e73c8409dfb751ac`, built per `configs/build-manifests/ds4-v41.json`.
  - The `ds4` and `score_official` hashes are recorded in `frozen-inputs.json`.
- Model: `DeepSeek-V4.1-Flash-Q2.gguf`, pinned in `configs/pins/ds41f-weights.json`. The full-file sha256 must match before any load.
- Fixture: ds4 `gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router/manifest.tsv`, 112 official-API cases.
  - The manifest hash is in `frozen-inputs.json`.
- Metal reference: `results/base-default.tsv` from the same fixture (upstream M5 Max, 48 GiB cache, NLL 0.396840).
  - The reference hash is in `frozen-inputs.json`.
  - Upstream names its file `DeepSeek-V4.1-Flash-IQ2_XXS-Q2_K-imatrix.gguf`; byte identity with the published Q2 is not established. A small delta may therefore mix a backend difference with an artifact difference.
- Scorer: `results/dsv41-flash-gates/harness/score_fidelity.py`. Its formulas and PASS rule are in the docstring and fixed at this commit.

## Arms
1. **Greedy text smoke.** Run:

   `ds4 --cuda -m Q2 --ssd-streaming --ssd-streaming-cache-experts 48gb -c 8192 --nothink --temp 0 -n 128 -p "Explain in three sentences why the sky is blue."`

   The cache is explicit (48 GB, the same as arm 2). Automatic sizing could exceed `MemoryMax`.

   PASS requires all of:
   - exit 0;
   - 128 tokens produced;
   - coherent on-topic text, reviewed by eye and saved verbatim.
2. **Fidelity.** Run:

   `score_official Q2 manifest.tsv cuda.tsv 34816 --ssd-streaming --ssd-streaming-cache-experts 48gb`

   Score it against the Metal reference with the fixed scorer.

## Containment (both arms)
- Wrapper: `glm_safe_run.sh`, run as the current user.
  - Kill floor: 40 GiB (cache-off-style probe floor).
  - Minimum start: 110 GiB.
  - Timeout: arm 1 900 s, arm 2 5400 s.
- Transient user unit `glm52-dsv41-smoke-*`. The wrapper requires the `glm52-` prefix; this is not a GLM run. Unit properties:
  - `MemorySwapMax=0`, `OOMPolicy=kill`, `KillMode=control-group`;
  - `MemoryHigh=72G`, `MemoryMax=76G`.
- Sizing: 48 GB expert cache (44.7 GiB), plus 8.1 GiB dense, plus runtime and file-cache headroom. Physical 119.7 GiB − 76 GiB leaves 43.7 GiB, which is at least the 40 GiB floor.
- Any OOM, cgroup kill, swap, Xid, short output, timeout or surviving descendant is a FAIL.
- Runner: `results/dsv41-flash-gates/harness/run_smoke.sh` (`text` \| `fidelity`).

## Candidate 2 revision (2026-09-25, after text attempt 2 FAILED)

These changes target only the two observed failures. The fidelity arm has not been run.

1. **Expert cache 48 GB → 42 GB in both arms.**
   - A 48 GB target makes ds4 allocate one 40.88 GiB dynamic cache. That reproducibly emits a kernel `NVRM NV_ERR_NO_MEMORY` line, seen in text attempt 2 and `diag-cache48gb`.
   - 4, 24, 36 and 42 GB targets loaded clean (`diag-cache*`).
2. **Text arm criterion.** The three-sentence prompt ends at EOS near 100 tokens, so "128 tokens produced" was unreachable. The arm now uses a long-form prompt with `-n 256`.
   - New PASS rule: exit 0, empty wrapper `kernel.log`, and at least 128 tokens produced (engine stops at `-n` or EOS; count verified by retokenizing the output with the model tokenizer through `ds4 --dump-tokens --raw`).
   - Coherent, on-topic text is saved verbatim and reviewed by eye.
   - Decode speed is not measured in this arm.
3. **Runner fix.** The kernel-event capture used `journalctl --since` with an unparseable timestamp. The wrapper's own `kernel.log` remains the authoritative Xid/OOM check.
