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

## Candidate 3 (2026-09-26, after the GPT-6 sol high review)

These changes close the review findings named below. No acceptance arm has run since candidate 2 was frozen. Candidate 2's arms were never executed, so candidate 3 replaces candidate 2.

Frozen inputs: `frozen-inputs-c3.json` (`artifact_sha256` executables, model identity, fixture, reference, scorer script and `component_sha256` harness hashes). The earlier `frozen-inputs.json` is kept for the record.

- **H1 (memory arithmetic).** The earlier "119.7 − 76 GiB ≥ 40 GiB" argument is **withdrawn**. CUDA allocations on GB10 are not charged to the memory cgroup: text attempt 2 peaked at 1.2 GB of cgroup memory while MemAvailable fell by about 63 GiB. `MemoryHigh` and `MemoryMax` remain as a host-side backstop only.
  - The binding safety control is the wrapper's system-wide MemAvailable kill floor of 40 GiB.
  - Admission bound: MemAvailable at start (≥ 110 GiB by the memory guard) minus ds4's planned total at the 42 GB target must be at least the 40 GiB floor. The planned total is 3.29 GiB context, 9.37 GiB resident, 34.87 GiB cache and 7.12 GiB prefill reserve, about 54.7 GiB; the text arm's -c 8192 differs only in KV. 110 − 54.7 = 55.3 GiB, which meets the floor. The fidelity arm's 34,816 context adds under 1 GiB of KV.
  - The low-point check is part of each attempt's `summary.json`.
- **H2 (identity).**
  - The runner verifies the executable's sha256 against `frozen-inputs-c3.json` and requires it to be read-only.
  - It verifies the model's size, inode and mtime against the record from its full-file sha256 verification, and requires the model to be read-only.
  - It does both before launch and after exit, and writes `identity.json`. An arm without a verified identity fails.
  - Limitation: the wrapper's continuous executed-binary provenance applies only to `ds4-server` under `~/.cache/glm52-*` and is not used here. Identity rests on read-only files plus before and after hashes.
- **H3 (scorer).**
  - The scorer validates IDs against the fixture manifest, integer and range constraints, and equal per-case coverage.
  - `harness/test_score_fidelity.py` holds 14 mutation tests. All pass, including the invented-ID and negative-NLL attacks.
  - The fixture manifest is now a required argument.
- **H5 (bundles).** `harness/bundle_attempt.py` writes `manifest.json`, `raw.jsonl` and `summary.json` for every attempt. The earlier attempts got after-the-fact bundles rebuilt from preserved logs.
- **H6 (token count).** The text arm adds `--dump-logprobs steps.json`. That records each engine-emitted greedy step's token id, text and bytes from `run_logprob_dump` in `ds4_cli.c` at 0aaea5a2.
  - PASS requires at least 128 steps, and the text reconstructed from token bytes is saved.
  - The retokenization rule from candidate 2 is withdrawn.
  - This remains a smoke check. Decode speed is not measured here.
- **Fidelity-arm working directory.** The fixture manifest uses paths relative to the ds4 checkout, so the unit now runs with `--working-directory=$SRC` and an absolute output directory.
- **M9.** A fidelity delta measures combined artifact and backend fidelity, because the Metal reference GGUF's byte identity is not established. This is labeled in `summary.json`.
- **L10.** The `glm52-` unit prefix stays for smoke only. The production profile needs its own validated unit name.
