# MiMo-V2.6-Flash M3: acceptance qualification, candidate 1 (preregistered)

This is the acceptance run of profile `mimo-v2.6-flash/cuda-spark-128g-262k`,
switch alias `mimo26flash`, at the commit that adds this file (the freeze).
It was committed before any weights were loaded for it.

- **Verdict:** `gate_verdict.py` in this directory writes `gate-summary.json`.
- **Kit bundle:** the qualification kit writes `manifest.json`,
  `summary.json`, `SUMMARY.md` and one directory per cell into this same
  directory.
- **On `PASS`:** M4 promotes the profile to `qualified` (test-first). The
  owner-authorized holdout draw then runs as its own preregistered run on
  this configuration.

MiMo becomes a switchable alias. `qwen38fn` stays the default (owner,
2026-10-12).

## Configuration under test

The profile at the freeze serves:
- the MikeRoz MOPD 2.52 bpw EXL3 pack (94.30 GiB) with the 4-bit DFlash
  drafter;
- exllamav3 1.6.1 `f45b8451` and TabbyAPI `2636ef60`, run through
  `scripts/lib/tabby_launch.py`;
- one slot of 262,144 tokens, Q8 cache;
- host-RAM caps: sliding-window checkpoints 512 MB, image embeddings 256 MB.

The M1 smoke (`../smoke-2026-10-13-r3`) and both M3 preflights
(`../preflight-2026-10-13`, `../preflight-2026-10-13-r2`) are diagnostic
history for this configuration, not evidence for this verdict.

## What runs (`run-c1.sh`, one fresh start)

**Refusals before anything starts.** Each logs `REFUSED` and starts nothing:
- The working tree must have no tracked edits and no uncommitted files here.
- `gate_verdict.py --preflight` must show the tree equal to the freeze under
  every `FROZEN` path.
- Then production is stopped through the switch, and this shell takes the
  system inference lock (fd 8). Every child starts with that fd closed, and
  the lock is released before the restore.
- `verify_identity.py` must find:
  - `src/tabbyAPI` and `src/exllamav3` at the build manifest's revision and
    tree, with no tracked edits;
  - the exllamav3 wheel at its pinned sha256, with every file it installs
    byte-identical in the runtime;
  - every served file of the pack and the drafter at the weights manifest's
    size and sha256.
- The page cache is released, and MemFree must reach 108 GiB for 3 samples.

**Launch.**
- `93_profile_serve.sh` runs in a user scope with the profile unit's
  `MemoryHigh=8G` and `MemoryMax=16G`, plus `MemorySwapMax=0`,
  `OOMPolicy=kill` and `RuntimeMaxSec=14400`.
- It arms the profile memwatch at 8 GiB.
- The pack's page cache is released every 1 s until ready.
- Observers: 1 Hz `/proc/meminfo` plus the scope's `anon` and `file`, and
  `journalctl -k`.

**After ready, in this order:**
1. `probe-c1.json`: three greedy prompts (`../smoke-2026-10-13/probe.py`).
2. `agent-gate.json`: `regression-suite.py agent-gate --engine vllm --model
   mimo-v2.6-flash --prefix-block-tokens 256`.
   - `vllm` selects the OpenAI-usage semantics: response reuse counts and
     client-measured TTFT.
   - 256 is one exllamav3 page. Prefill stashes a sliding-window checkpoint
     at the last full prompt page, so a healthy repeat re-runs under one page
     (the preflights measured it).
3. The kit, `--cells speed,toolcall,vision,context --with-soak`:
   - **speed:** the block-aware bench, 8 tokens per event at most
     (`speed_max_tokens_per_event`), at context 0 and on the 28,672-token
     fixture;
   - **tool-call:** 20 cases;
   - **vision:** MMMU-val-100, chat mode, a 4,096-token answer budget (the
     profile's `vision_max_tokens`);
   - **context:** the direct fill of one slot × 250,128 tokens (cap 262,144)
     with needles and controls, floor 8 GiB;
   - **soak:** 30 minutes at the repository's 5 GiB floor.

Then the server stops, the scope must empty, and `qwen38fn` is restored.

## Gate formula (fixed)

`PASS` requires every clause. Any failed clause gives `FAIL`, whatever else is
missing.

Otherwise, any of the following gives `NO_RESULT`:
- missing or malformed evidence;
- a `FAILED_ATTEMPT` line;
- a log without the line proving it saw the run (`session start`, `ARMED`),
  or a driver log without `kernel journal alive at exit`;
- code or artifact identity that differs from the freeze;
- a kit manifest whose git head is not the driver's;
- samples that do not cover the launch to the empty scope at most 5 s apart.

1. **No kill lines**, by the M1 smoke scorer's patterns:
   - kernel: `out of memory|oom-kill|oom_reaper|NVRM|Xid`, case-insensitive;
   - server: `CUDA out of memory|OutOfMemoryError|NVRM|Xid`;
   - memwatch: `BREACH|EMERGENCY_STOP|FAIL_CLOSED`.
2. **Swap growth ≤ 0.25 GiB** over the run.
3. **External low point ≥ 8 GiB** MemAvailable, from launch to the empty
   scope.
4. **Peak use ≤ 106 GiB, and the switch admission `ceil(peak + 8)` ≤ 115.**
   Peak use is the largest MemAvailable between the MemFree gate and the
   launch, minus the low point.
5. **Direct 262K:** the context verdict is `PASS`, with ≥ 250,000 server
   prompt tokens and the kit's resolved cap exactly 262,144.
6. **Text identity:** all three greedy hashes equal preflight r2's
   (`../preflight-2026-10-13-r2/probe.json`).
7. **Tool-call ≥ 19/20.**
8. **Agent gate:** all four tests pass, with the semantics above.
9. **Soak:** `pass`, held to the default 5 GiB floor.
10. **Vision:** 100 cases, no errors, and not significantly worse than the
    default's 64/100 (Flash-Next c10). The one-sided pooled two-proportion
    z must be ≥ −1.645, so at least 53 correct.
11. **Speed suite valid:** both levels valid. A valid suite is what makes the
    reported numbers publishable; the values themselves do not gate.

`gate_mutations.py` builds a clean synthetic bundle and 41 broken or failing
variants. The scorer must give each its expected verdict (42/42 before the
freeze).

## Reported, not gating

Each is reported beside Flash-Next's c10 kit value.

- **Decode at 0 and 28,672, prefill, TTFT, and the 262K TTFT.**
  - Flash-Next's c10 values: 34.07 / 40.92 tok/s, 2,711 tok/s and 0.21 s.
  - Speed targets are general; misses are reported (owner, 2026-09-11).
  - On `PASS` these are MiMo's only publishable speed numbers.
- **Fidelity:** M1's measurement against the release weights: ΔNLL 0.0503,
  KLD 0.0836, top-1 −0.94 pp. It is reported, not waived; the owner decides
  on any adoption.
- **Accuracy:** the separate holdout run.
- **Kit verdict:** `measured` (the profile declares no targets).

## Freeze and randomness

- **Freeze.** The commit that adds this file freezes these:
  - the profile, the build and weights manifests, and the host file;
  - `scripts/lib`, the composed kit scripts, the fixtures and the MMMU and
    tool-call evalsets;
  - the probe, the smoke scorer and preflight r2's probe;
  - this driver, the scorer, the mutations and the identity check.

  After the start, any change to these is a new candidate in a new
  directory.
- **No public randomness is drawn.** There is one arm and no fixture choice:
  the kit's fixtures are fixed by its scripts (speed seed 42, the fixed
  MMMU-val-100 and tool-call sets, deterministic context needles). This
  follows the precedent of Flash-Next's first qualification
  (`results/qwen38fn-gates/qualify-2026-10-11/PREREG.md`) and the GLM-5.3
  bundle. The plan's "drand after the freeze sets the order" has nothing to
  order here.

## Review

Two persistent reviewers review the frozen candidate. Reviewers choose their
own severities. The candidate is run only when no verified high or critical
finding is open. The convergence rule is AGENTS.md's.
