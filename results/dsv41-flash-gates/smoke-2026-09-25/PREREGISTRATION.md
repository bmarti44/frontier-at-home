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

## Candidate 4 (2026-09-26, after GPT-6 sol high review round 2)

Round 2 left 3 HIGH findings, down from 6. It also left H3 partially closed, H5 partially closed and L10 open (disclosed). These changes close only those items. No acceptance arm ran under candidate 3.

Frozen inputs: `frozen-inputs-c4.json`. It adds `test_bundle_attempt.py`.

1. **Bundler false-PASS paths.**
   - Fixed required artifacts per arm. Missing means FAIL, never an empty default.
   - Every required artifact must be written at or after `started_at`.
   - The runner refuses a non-empty output directory (exit 4).
   - Fidelity `raw.jsonl` holds every paired case row from `cuda.tsv`.
2. **Text evidence.** The bundler rebuilds `output.txt` from the engine-emitted token bytes. The text arm is `PENDING_REVIEW` (bundler exit 2) until `text-review.json` records `coherent_on_topic`, written after reading `output.txt`. A false value is a FAIL.
3. **Swallowed failures.** The runner exits with the bundled verdict: 0 PASS, 1 FAIL, 2 PENDING_REVIEW, 3 bundler error. The engine status stays in `exit_code.txt`. The scorer's exit status and log are kept (`fidelity-score-exit.txt`, `fidelity-score.log`).
4. **H3 remainder.** Before launch, the runner verifies the frozen hashes of the harness scripts, the fixed scorer script, the fixture manifest and the Metal reference TSV (`frozen-check.json`; mismatch is exit 22).

`harness/test_bundle_attempt.py` holds 15 mutation tests covering these paths. All pass. The post-hoc bundles of earlier attempts were regenerated with this bundler and remain FAIL.

## Candidate 5 (2026-09-26, after GPT-6 sol high review round 3)

Round 3 found 3 HIGH findings and 1 MEDIUM. Each narrows a round-2 item; none is a new finding class. The count stayed at 3. No acceptance arm ran under candidate 4. These changes close only those findings.

Frozen inputs: `frozen-inputs-c5.json`.

1. **Fidelity bound to a successful, reproducible score (round 3, finding 1).**
   - The runner copies the Metal reference and fixture manifest into the attempt (`metal-reference.tsv`, `fixture-manifest.tsv`) and scores from those copies.
   - The bundler requires:
     - scorer exit status 0;
     - the bundled copies and the scorer at their frozen hashes;
     - a re-run of the frozen scorer on the bundled `cuda.tsv` and copies that reproduces `fidelity-summary.json` exactly, ignoring only its three path fields.
   - `raw.jsonl` holds both arms' case rows.
2. **Text review bound to the output (finding 2).** `text-review.json` must record `artifact_sha256["output.txt"]`, and it must equal the hash of the `output.txt` the bundler reconstructs from the engine's token bytes. Otherwise the result is FAIL.
3. **Frozen inputs verified at use (finding 3).** The frozen-input check runs at launch (a mismatch aborts, exit 22), after the run, and after scoring. The bundler requires every expected check to be present and `ok`, and records them in `manifest.json`.
4. **Timestamps (finding 4, MEDIUM).** `started_at` and `finished_at` must be full ISO timestamps with a UTC offset; anything else is FAIL. Required artifacts must be written after the start. Engine-written artifacts (containment logs, `steps.json`, `cuda.tsv`) must not be newer than `finished_at`. Both bounds have a 1 s tolerance.

`harness/test_bundle_attempt.py` has 12 new mutation tests, 27 in total. All 12 fail against the candidate-4 bundler and pass on candidate 5. The derived post-hoc bundles were regenerated and remain FAIL.
