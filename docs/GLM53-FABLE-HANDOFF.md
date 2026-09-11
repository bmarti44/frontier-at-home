# Prompt for Fable: review, plan, and finish GLM on the Spark

You are taking over an unfinished integration in `/home/bmarti44/spark-deepseek-v4-flash` on my DGX Spark. Review the actual repository and evidence, make a short practical plan, then execute it. Do not stop after giving me a plan. This has already taken days; prioritize a working, useful result and avoid another open-ended series of experiments.

## What I want

Get GLM-5.3-Flash running reliably and at a decent speed for agentic work through this repo's named profiles. Speed matters more to me than maximum fidelity, but report measured quality tradeoffs honestly and respect the repository's existing acceptance requirements. Keep GLM optional; do not make it the default.

The full target is **1,048,576 tokens TOTAL across four simultaneous slots of 262,144 tokens each**, plus text, tools, images and up to 16-frame video. Keep authentication, existing default, switching safeguards and rollback intact. A smaller working development profile is useful, but does not complete the full qualification goal.

## Start from these facts; verify them

Handoff prepared September 10, 2026, about 8:13 p.m. Eastern.

- Branch: `claim-model/glm-5.3-flash/cuda`. Draft PR: https://github.com/bmarti44/frontier-at-home/pull/6. Latest implementation/preparation commit before this handoff: `1417c0fd`; check HEAD for the subsequent handoff commit.
- GLM has run and answered requests on this machine. Both named experimental profiles are implemented. The production switch remains gated; do not equate an executable experimental profile with production approval.
- `configs/profiles/glm-5.3-flash/cuda-spark-128g-agent-fast.json`: four 65,536-token slots, batch512, 4GiB KV reservation.
- `configs/profiles/glm-5.3-flash/cuda-spark-128g-1m-experimental.json`: four262,144-token slots, batch128, long-prefill threshold32, KV9,565,304,320 bytes. Uses existing quantized weights and FP8 cache. Only batching was changed in the latest fallback; numerical code and safety limits were preserved. Profile tests and focused review passed.
- Historical `context-direct-007` passed four250,128-token inputs under128/32. Later512/128 settings failed the18GiB memory reserve in `context-direct-010`. Preserve both results; the historical pass does not qualify today's exact configuration.
- The latest `context-direct-011` prepared fresh inputs and successfully started the current128/32 profile with authenticated READY. The owner requested this handoff **before the short check or million-token requests ran**. It is NO_RESULT for context qualification, not a new full-context pass.
- That server was stopped safely for handoff. Launcher/stop exits were0, identity guardPASS, no model descendants or cgroup remained, and memory recovered above110GiB. All120 frozen file bindings and2,486 prepared cache files still matched. Full post-run model/runtime inventory verification is still pending; prefreeze and launch checks passed.
- Qwen `qwen38-1m` remains the recorded/reboot default. A recorded default does not establish a live production backend; inspect actual health. Proxy and guard configuration were unchanged.
- Swap is temporarily disabled by the owner. Docker, its socket and containerd were also stopped by the owner. Preserve and explicitly account for this temporary host state; do not leave it undocumented.
- Scoped passwordless runtime control is installed. Use the delegated operations; do not ask me to repeat sudo setup or broaden privileges.

## Read these first

1. `AGENTS.md`, `docs/GLM53-QUICKSTART.md`, and `results/glm53-flash-gates/STATUS.md`.
2. `results/glm53-flash-gates/context-scheduler-128-001/PROTOCOL.md` and its tests, review and freeze recipe.
3. `results/glm53-flash-gates/context-direct-011/README.md` and handoff records.
4. Local latest evidence under `~/.cache/glm53-flash/`: `qualification-freeze-004`, `context-direct-011`, `context-direct-011-observations`, and `server-20260910-200747`. Keep API keys private. These contain the original inputs, timestamps, launch and stop records; full archive publication is not finished.
5. Closed negative results `context-direct-010` and `indexer-workspace-preparation-001`; native reference progress `bf16-one-layer-005`.

## What remains

- Fresh confirmation of today's full-context settings, then sustained reliability with the required fresh-server workload.
- The fixed100-case paired fidelity comparison. Ollama `kimi-k3:cloud` generated100 test prompts, **not native GLM reference probabilities**. Whole native BF16 GLM does not fit this Spark. One real KDA layer passed a contained native probe; a full streamed reference has not been built. Review feasibility and cost before starting more kernel or reference machinery. Do not substitute Kimi answers for BF16 logits or claim this requirement passed.
- Maximum-media checks: smaller224-size fixtures passed; the intended512-size limits remain unqualified.
- Production switching/rollback/lifecycle gates and final review.
- Properly qualified speed measurements. There is no accepted headline performance figure yet; do not advertise diagnostic or short-test throughput as production speed.

## How to finish efficiently

First give me a plain-language review: what works, the actual blockers, and what previous work can be reused. Then state a short ordered plan and carry it out. Choose the smallest justified changes; reuse existing launchers, safety wrappers, fixtures and scorers. Do not rebuild closed harnesses, repeat signed-off component reviews, introduce new profile variants without need, or retry a failed experiment unchanged.

For the128/32 direct test, reuse `scripts/48_probe_glm53_context.py`, `context-native-profile-002/bind_launch.py`, and `context-clear-instruction-001/{prepare_inputs.py,run_short.py}`. The512/128 `context-scheduler-003/adapter.py` is not the validator for this profile. Freeze the actual selected sources and obtain fresh verifiable randomness for changed/new candidates. Preserve direct011 as an interrupted attempt rather than overwriting it.

Only one large model may run. Follow AGENTS.md's inference lock, hardened wrapper,110GiB start requirement, current92/94GiB containment and18GiB kill floor. Never lower a safety or quality gate to manufacture a pass. Complete auxiliary publication/build work before a loaded-model test. Reuse the two existing persistent reviewers if available.

The operator commands already exist:

```bash
PATH=/usr/bin:/bin scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-agent-fast start
PATH=/usr/bin:/bin scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental start
PATH=/usr/bin:/bin scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental status
PATH=/usr/bin:/bin scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental stop
scripts/52_engine_switch.sh status --json
```

Use one profile at a time and verify current state before starting anything. Keep commits small, evidence reproducible, the PR current, and progress updates understandable. Finish with exact run instructions, measured results, and an honest list of any unmet requirements. If hardware makes a requirement infeasible, establish and explain that concrete limit promptly; do not silently narrow the goal or spend days disguising the gap with supporting experiments.
