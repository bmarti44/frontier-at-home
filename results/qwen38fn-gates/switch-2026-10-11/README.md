# Production switch: qwen38fn in, qwen38-1m back, stop (2026-10-11)

The owner chose Qwen3.8-Flash-Next as an optional engine (alias `qwen38fn`),
with `qwen38-1m` staying the default. This run took the production switch
through the new alias and back again.

How it ran:
- The production checkout was on `main` at `10995023` (PRs #11 and #12 merged).
- Only the owner-installed NOPASSWD grant was used
  (`sudo -n scripts/52_engine_switch.sh <alias>`); no new sudo setup.
- arlab granted the GPU and held all load throughout.
- Driver: `run-switch.sh`. Full transcript: `switch-log.txt`.
- `observe/` holds:
  - the 1 Hz host memory log;
  - the kernel log;
  - the switch's server and memwatch logs for `qwen38fn`;
  - `nvidia-smi` while serving.

| Step | Result |
|---|---|
| `status --json` before | `qwen38-1m` recorded; no engine running |
| `sudo -n scripts/52_engine_switch.sh qwen38fn` | **exit 0 in 623 s** (see below) |
| `status --json` after | `qwen38fn` recorded |
| `sudo -n scripts/52_engine_switch.sh qwen38-1m` | **exit 0 in 59 s**. The memory guard read 115.5 GiB against 100 required. The switch verified the Qwen hashes first, then stopped `qwen38fn` (memwatch `DISARMED`), started `qwen38-engine` and verified it. The default was restored. |
| `sudo -n scripts/52_engine_switch.sh stop` | **exit 0 in 2 s**. `active.json` is unchanged (`qwen38-1m`). |
| End state | No engine running: `qwen38fn-engine` and `qwen38-engine` both inactive. 115.44 GiB available. |

The `qwen38fn` step in detail:
- The memory guard read 115.3 GiB against the 113 required.
- The switch ran its digest checks, then the `/health` and model-card checks.
- The unauthenticated `8010/health` returned 401.
- The authenticated semantic probe answered `**4**` (finish `stop`, 27 tokens, with `reasoning_content` present).
- `/v1/models` on 8013 lists `qwen3.8-flash-next` with `max_model_len` 262144.
- vLLM loaded 75.65 GiB of weights and reported a KV cache of 1,079,117 tokens at 262,144, the same as candidate 5.

## Memory and kernel

- **Host MemAvailable low:** 12.54 GiB (13:25:24Z). This came during step 2's
  Qwen hash walk, while `qwen38fn` was still serving. The low during step 1
  was 13.00 GiB.
- **SwapFree:** 15.52 GiB at the start and 15.51 GiB at the end (minimum 15.49).
- **NVRM:** the kernel logged one line, at 13:26:11Z:
  `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal`.
  It came during **step 2, the `qwen38-1m` (llama.cpp) load**:
  - `qwen38fn` had fully stopped by 13:25:50, when MemAvailable was back to 121 GiB.
  - `qwen38-engine` started at 13:25:53.
  - At 13:26:11, MemFree was 2.7 GiB (0.74 GiB minimum at 13:26:16) behind
    about 21 GiB of page cache, while MemAvailable was 22–25 GiB.
  - The load completed and passed every switch check.
- **No other kernel lines:** no Xid, OOM or cgroup kill, and the `qwen38fn`
  step logged no kernel line at all.

The repo treats any NVRM line as a failure, so this is recorded as an **open
item against the default engine's load path**, not the new alias. The three
earlier `qwen38-engine` starts in the kernel journal (Oct 1, 2 and 10, all
boot restores from a quiet host) logged no NVRM line nearby. This start came
right after another engine released about 100 GiB, and right after the
switch's hash walk had read the Qwen GGUF into the page cache. Low MemFree
behind reclaimable cache is the same pattern that produced Flash-Next's
startup NVRM lines (`../diag-drafter-2026-10-11/`). There, evicting the
checkpoint's page cache during startup removed them. This run does not test
whether the same fix applies to llama.cpp.
