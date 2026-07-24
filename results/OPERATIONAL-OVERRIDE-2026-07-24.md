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
