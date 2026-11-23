# Inkling 2 Mini M3: acceptance qualification, candidate 2 (preregistered)

This is the acceptance run of profile `inkling-2-mini/cuda-spark-128g-1m`, the future switch alias `inkling2mini`. It runs at the commit that last changes this file (the freeze). This file was committed before any weights were loaded for this candidate.

**Candidate history.** Candidate 1 (`../qualify-2026-11-23-c1`, frozen at `382e7a8c`) never ran. Checked against the screen carrier's real evidence after its freeze, its clause 11 read a `metadata["method"]` key that the block-aware bench does not write (it writes `decode_method`), so every real run would have scored `NO_RESULT`. The mutation suite did not catch it, because its synthetic bundle used the same wrong key. This candidate changes only that check, the mutations' speed metadata, a real-evidence check in the mutation suite, the abort text (reviews B7 and B8) and its own names.

- **Verdict:** `gate_verdict.py` in this directory writes `gate-summary.json`.
- **Kit bundle:** the qualification kit writes `manifest.json`, `summary.json`, `SUMMARY.md` and one directory per cell into this same directory.
- **On `PASS`:** M4 promotes the profile to `qualified`, test-first, and adds the alias. The owner-authorized holdout draw then runs as its own preregistered run on this exact configuration (M5). The promotion verdict follows from that draw.

## Configuration under test

At the freeze, the profile serves:
- the official checkpoint `thinkingmachines/Inkling-2-Mini` at `b85c3cfa` (NVFP4/MXFP8 QAT experts, as shipped);
- vLLM nightly `0.34.0rc1.dev1133+g647a76866` with patches 0001–0004 (`vllm_tree` pinned in the build manifest);
- four native 262,144-token slots (1,048,576 total), BF16 KV in a fixed 68e9-byte arena, prefix cache;
- images on (4 per prompt, up to 4096 px a side), video off;
- `HostCacheReleaseWorker` with the Inkling and Flash-Next runtime roots;
- the lever screen's carrier levers: **DSpark speculative decoding x7 (probabilistic draft sampling), `--moe-backend humming` and `INKLING2_MAX_CTAS_POW2=1` (vLLM PR #69288), at the unchanged 2,048-token batch** (`../tune-screen-2026-11-23/summary.json`, carrier `a6`);
- reasoning effort **low** (`../tune-screen-2026-11-23/effort-a6/pick.json`).

The smoke runs (`../smoke-2026-11-23`) and the lever screen are diagnostic history, not evidence for this verdict.

## What runs (`run-c2.sh`, one fresh start)

**Refusals before anything starts.** Each logs `REFUSED` and starts nothing:
- The working tree must have no tracked edits and no uncommitted files here.
- `gate_verdict.py --preflight` must show the tree equal to the freeze under every `FROZEN` path.
- The agent gate's fixture, which `regression-suite.py` reads from the production checkout, must be byte-identical to the frozen `fixtures/ctx-32k.txt`. The driver logs its sha256 prefix.
- No other window may be running: nothing listens on :8018 and no `inkling2-*` user unit is active.
- Then production is stopped through the switch, and this shell takes the system inference lock (fd 8). Every child starts with that fd closed, and the lock is released before the restore.
- `verify_identity.py` must find:
  - the runtime's `python3.12` at its pinned sha256;
  - `site-packages/vllm` at the manifest's `vllm_tree` (the wheel plus exactly patches 0001–0004);
  - every patch file at its pinned sha256;
  - `uv pip freeze` of the runtime at the build manifest's `freeze_sha256` (torch, triton, flashinfer and `humming_kernels` at their installed versions);
  - every weight file at the weights manifest's size and sha256, and no other file outside the downloader's `.cache/`.
- The page cache is released. MemFree must reach 100 GiB and MemAvailable 110 GiB on the same 3 consecutive samples, within 600 s.

**Launch.**
- `93_profile_serve.sh` runs in a user scope with `MemoryHigh=92G`, `MemoryMax=96G`, `MemorySwapMax=0`, `OOMPolicy=kill` and `RuntimeMaxSec=21600`.
- It arms the profile memwatch at 10 GiB.
- Observers: a 1 Hz `/proc/meminfo` sampler plus the scope's `anon` and `file`, and `journalctl -k` from the driver's start to 5 s after the engine stop. The restore's kernel log is recorded in `kernel-restore.log` and is not scored.
- The Triton disk cache is left as the screen's last start left it, as a production restart would find it.

**After ready, in this order:**
1. `probe-c2.json`: three greedy prompts (`../smoke-2026-11-23/probe.py` through `probe_low.py`, rendered at effort low; clause 6).
2. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model inkling-2-mini --prefix-block-tokens 64 --spec-block-drop`. vLLM caches whole blocks of its KV groups' lcm, which the startup log gives as `kv lcm block sizes 64`.
3. The kit, `--cells speed,toolcall,vision,context --with-soak`:
   - **speed:** contexts 0 and 28,672, block-aware at the profile's K;
   - **tool-call:** 20 cases;
   - **vision:** MMMU-val-100, chat mode, a 4,096-token answer budget (the profile's `vision_max_tokens`);
   - **context:** the direct fill of 4 slots × 250,128 tokens (cap 1,048,576), with needles and controls, at the profile's 10 GiB floor;
   - **soak:** 30 minutes at the repository's 5 GiB floor.

Then the server stops, the scope must empty, and `qwen38fn` is restored in its own session. Its output stays outside the evidence until it is copied in.

**Launch and abort.** The driver is launched with `setsid -f` and stdout to a file, never a pipe. To abort, stop the scope (`systemctl --user stop inkling2-c2-<driver pid>.scope`): the cells fail fast, and the driver stops, closes the journal and restores production. The driver is never SIGKILLed, because bash defers a trapped signal until the running step returns. If the soak is running, `pkill -TERM -f scripts/35_soak.py` follows the scope stop, because the soak client retries until its 30-minute deadline after the engine dies (review B7). An abort is recorded in `ATTEMPTS.md` as an operator abort, not a model failure; memwatch's `EMERGENCY_STOP` line still fails clause 1, so an aborted run never passes (review B8).

## Gate formula (fixed)

`PASS` requires every clause. Any failed clause gives `FAIL`, whatever else is missing.

Otherwise, any of the following gives `NO_RESULT`:
- missing or malformed evidence;
- a `FAILED_ATTEMPT` line;
- a log without the line proving it saw the run (`session start`, an `ARMED target_pid=` line), or a driver log without `kernel journal alive after the engine stop`;
- code or artifact identity that differs from the freeze, including the agent gate's fixture;
- a start gate line below MemFree 100 / MemAvailable 110 GiB;
- a scorer or reference probe that is not the freeze's, or the mutation overrides used outside `--mutation`;
- a kit manifest whose git head is not the driver's;
- samples that do not cover the launch to the empty scope at most 5 s apart, or whose last sample at or before the launch is more than 5 s before it;
- a server log that does not show exactly the configuration under test:
  - the carrier's lever lines, and no other lever's;
  - `'reasoning_effort': 'low'` in the non-default args;
  - patch 0004's `INKLING2_MAX_IMAGE_SIDE=4096` line;
- an invalid speed suite or level, or one not at the profile's K (clause 11). Its numbers are unmeasured, not failed.

The clauses:

1. **No kill lines**, with the screen's patterns:
   - kernel log, from the driver's start to 5 s after the engine stop: `out of memory|oom-kill|oom_reaper|NVRM|Xid`, case-insensitive. The window includes production's stop, so a kill line there also fails this candidate (conservative, as in earlier gates);
   - server log: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
   - memwatch log: `BREACH|EMERGENCY_STOP|FAIL_CLOSED`.
2. **Swap growth ≤ 0.25 GiB.** This is the most swap used from the launch to the empty scope, minus the swap used at the last sample at or before the launch.
3. **External low point ≥ 10 GiB MemAvailable**, from the launch to the empty scope. This is AGENTS.md's direct-1M floor.
4. **Switch admission `ceil(peak + 10)` ≤ 115 GiB.** Peak use is the launch baseline minus the low point. The launch baseline is the largest MemAvailable between the MemFree gate and the launch, counting the gate line's own value.
5. **Direct 1M:**
   - vLLM's KV capacity at 262,144 tokens per request is ≥ 1,048,576 tokens;
   - the context verdict is `PASS`, with ≥ 1,000,000 actual input tokens;
   - the kit's resolved cap is exactly 1,048,576.
6. **Text path identity (gating).** The three greedy hashes must equal the screen carrier's (`../tune-screen-2026-11-23/a6/probe.json`), or the clause fails. `probe_low.py` runs the smoke probe unchanged except that every request carries `chat_template_kwargs: {"reasoning_effort": "low"}`. vLLM merges that over the server default, so the rendered prompt is a6's, which ran at low by default, whatever effort this candidate serves. Greedy output has been start-to-start identical in the screen: a0 = a0r across two windows, and a4 = a6 with DSpark on.
7. **Tool-call ≥ 19/20.**
8. **Agent gate:** all four tests pass, with the semantics above, each reported exactly once.
9. **Soak:** `pass`, held to the default 5 GiB floor.
10. **Vision:** 100 cases, no errors, and not significantly worse than the default's 64/100 (Flash-Next c10). The one-sided pooled two-proportion z must be ≥ −1.645, so at least 53 correct.
    - Inkling is measured at the 4,096-token answer budget. The default's 64 was measured at 256, with 3 invalid. The result reports this beside the comparison.
    - The promotion rule's stricter MMMU ≥ 64 is applied in M5, not here.
11. **Speed suite valid.** The suite and both levels must be valid, at the profile's `speed_max_tokens_per_event`, with the bench's block-aware method: `metadata.decode_method` must equal `block-aware: (client_completion_tokens - max_tokens_per_event) / (t_last_event - t_first_event)`. The values themselves do not gate here. They are this profile's publishable speed numbers, and the M5 promotion rule compares them with Flash-Next's.
12. **The host page-cache release worker ran:**
    - one runtime release line under each of `~/.cache/inkling-2-mini` and `~/.cache/qwen38-flash-next`, each with at least one file;
    - one model-root release line under `~/models` before the arena;
    - no `refusing to allocate the KV arena` line, which is `FAIL`.

`gate_mutations.py` builds a clean synthetic bundle and broken, failing or isolating variants: 73 cases, MiMo c1's 58 plus 15 for Inkling's clauses, carrier and identity checks. The scorer must give each its expected verdict before the freeze. Because a synthetic bundle shares the scorer's assumptions about field names, the suite also runs the readers on screen carrier a6's real logs, `speed.json` and `toolcall.json`, which must give no finding (candidate 1's defect).

## Reported, not gating

All are reported beside Flash-Next's c10 kit values (decode 34.07 / 40.92 tok/s, prefill @28,672 2,711.5 tok/s, TTFT 0.21 s):
- decode at 0 and 28,672, prefill, and TTFT;
- the 1M fill's TTFT and its low point;
- the DSpark acceptance length;
- the probe hashes against the screen carrier's and against a0's;
- load time, peak use and the admission value M4 uses for `minimum_start_gib`.

Fidelity: the weights are the release checkpoint as shipped, and the KV is BF16. The lossless levers were screened under PREREG rules (byte-identical probes for humming and #69288). Accuracy is measured by the M5 holdout on this configuration.

## Freeze and randomness

- **Freeze.** The commit that last changes this file freezes:
  - the profiles directory, the build and weights manifests, the host file and `patches/inkling2`;
  - `scripts/lib`, the composed kit scripts, the fixtures, and the MMMU and tool-call evalsets;
  - the probe, the smoke host file and the screen carrier's probe;
  - this driver, the scorer, the mutations, the identity check and `probe_low.py`;
  - the screen's `summary.json` and `effort-a6/pick.json`, which fix the carrier and the effort.

  After the start, any change to these is a new candidate in a new directory.
- **No public randomness is drawn.** There is one arm and no fixture choice. The kit's fixtures are fixed by its scripts: speed seed 42, the fixed MMMU-val-100 and tool-call sets, and deterministic context needles. This follows Flash-Next's first qualification and MiMo's c1.

## Review

Two persistent reviewers review the frozen candidate, and they choose their own severities. The candidate runs only when no verified high or critical finding is open. The convergence rule is AGENTS.md's.
