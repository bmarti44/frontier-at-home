# Operational override — 2026-07-24 (dev-grade ds4 v0.4.2 on port 8011)

**User-directed exception to the qualified serving posture.** The benchmark
record (results/DECISION.md) and the 2026-07-17 product override
(results/DECISION-OVERRIDE.md) are NOT modified by this note.

- What: engine port 8011 serves a DEV build of ds4 v0.4.2
  (`Entrpi/ds4` tag v0.4.2, commit `6a94bc7`, built today at
  `/home/dsv4/ds4-project/src/ds4-v0.4.2`, `make cuda CUDA_ARCH=sm_121`).
- Command: `ds4-server --cuda -m <IQ2XXS base> --host 127.0.0.1 --port 8011
  -c 65536 --kv-disk-dir /home/dsv4/ds4-v042-kvdisk` with
  `DS4_BATCH_FIT_HEADROOM_MB=16384`, run via nohup as dsv4 (no systemd).
- Why: measured wins for the agent workload (docs/ds4-v042-eval-2026-07-24.md):
  0.8-1.1 s warm turns with disk persistence, 1.6x prefill, +35% depth decode.
- Known gaps vs protocol: no accuracy qualification on this build, no
  build/weights manifest verification, not systemd-managed.
- Compensating controls (2026-07-24): startup warmup pre-pays the ~11 GiB
  lazy-init pool (steady free 12.0 GiB); a dedicated memwatch is ARMED at
  a 10 GiB kill line (`/run/dsv4/ds4-dev.target`,
  `/home/dsv4/logs/memwatch-ds4-dev.log`); exposure chain re-verified PASS
  (42_verify_exposure.sh; a stale ComfyUI tailnet route on :8443 was found
  and removed during this check); disk-KV tier bounded by the engine's
  4096 MiB default budget at `/home/dsv4/ds4-v042-kvdisk` (observed 1.5 GiB).
- Reboot behavior (intentional): the systemd unit still starts the QUALIFIED
  llama.cpp stack — an unattended reboot reverts to the safe engine.
- Rollback (tested repeatedly today): kill the ds4 process, then
  `21_serve_llamacpp.sh start` with the tuned env (or `systemctl start
  deepseek-v4-flash-llamacpp`); verify 401 on 8010/tailnet.
- Exit criteria: either full qualification promotes a manifested v0.4.x
  build (task: golden/parity/accuracy/soak + unit + guard mapping), or the
  experiment ends and llama.cpp resumes 8011.
- Owner: Brian. Review trigger: any memwatch BREACH, accuracy complaint,
  or 7 days elapsed.

## Update (same day, post sol memory analysis)

Serving env now adds `DS4_SERVER_COALESCE_MAX_TOKENS=2048` and
`DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1` (sol-identified reductions, no
code changes): first-generation init pool 11.9 -> 8.5 GiB, steady free
13.7 GiB (above the 12 GiB reference line, desktop apps running),
speculation intact (JSON 38.6 tok/s @ 95.7% acceptance), and the disk-KV
tier demonstrated cross-restart warm restore (19K prompt served at 0.4 s
TTFT by a freshly started process). Dev watchdog re-armed on the new pid.
Prefill cost of the coalesce cap on genuinely novel long prompts is not
yet quantified (the 19K probe was served from disk cache) — measure before
promoting these knobs into any qualified config. Full analysis:
docs/ds4-v042-memory-analysis-sol-2026-07-24.md (tail of the sol run;
ranked patch set including the conditional 5.5-6.3 GiB serial-fallback
guard and a ~3 GiB soft idle-release hook design for upstream).

## Decision (2026-07-24): idle-release not pursued

The soft idle-release design (docs/ds4-v042-memory-analysis-sol-2026-07-24.md
§3) is documented but intentionally NOT implemented: active-use memory is
already inside the safe envelope with the deployed knobs (steady free 13.7
GiB), the ~3 GiB it would recover only matters while the model is idle, and
the recurring multi-GiB free/realloc cycles carry a unified-memory
fragmentation risk. Revisit only if the box's between-conversation workloads
(training runs, etc.) start losing to the reserved workspace.

## Rollback of the memory knobs (same day, ~14:15)

Both sol-identified env knobs are REVERTED after live failure evidence:
- `DS4_SERVER_COALESCE_MAX_TOKENS=2048`: the first genuinely novel ~20K
  prompt (the user's agent preamble) hung the GPU worker in a 99.9%-CPU
  zero-progress loop for 10+ minutes with no log output — consistent with
  the cap (2048) sitting below the engine's prefill chunk (4096). The
  earlier "validation" passed only because its 19K probe was served from
  the disk-KV cache; the unquantified-novel-prefill gap noted at deploy
  time was the exact hole.
- `DS4_CUDA_NO_ATTENTION_OUTPUT_F16_CACHE=1`: with the hang cleared, a
  novel 19K prompt prefilled at 123 tok/s vs 776 tok/s with the cache
  enabled — the F16 attention-output cache is a ~6x prefill accelerator,
  not an idle 2.7 GiB.
Serving env is back to the gauntlet-validated original
(`DS4_BATCH_FIT_HEADROOM_MB=16384` only + warmup + disk-KV). Verified after
revert: novel 18K prompt at 776 tok/s prefill / 20 s TTFT; steady free
~10.6-12 GiB (warm banks included); dev watchdog re-armed on the verified
engine pid (earlier arms twice targeted a transient/sudo wrapper pid —
procedure now: resolve the engine as the child of the sudo wrapper).
Lesson recorded: never promote a serving-knob change without a
NOVEL-prompt probe; sol's rank table entries 2 and 3 are rejected on this
host by measurement.
