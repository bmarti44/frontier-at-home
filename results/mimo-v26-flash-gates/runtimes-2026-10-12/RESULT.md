# MiMo-V2.6-Flash runtimes built (plan B0 criterion 3); restore needed an owner cache drop

On 2026-10-12 the owner asked for the runtimes that the [readiness check](../readiness-2026-10-12/RESULT.md) found missing to be built. Both are now built and verified on the Spark. B0 stays **`NO_GO (readiness)`**: criteria 1 and 2, a fitting quant with published fidelity, still fail. No MiMo weights were downloaded and no MiMo model was loaded.

**Criterion 3 (a runtime we can build safely on GB10) is now met on both paths:**

| Runtime | Source | Build | Verified | Record |
|---|---|---|---|---|
| llama.cpp | b11514 `de7fa0a3`, the repository's `portable` pin. It has MiMo-V2.6 conversion (#29257) and the MiMo DFlash drafter (#29650). | `scripts/14_build_llamacpp_pin.sh --pin llamacpp-b11514-de7fa0a3 --host-class cuda-spark`. Clean build, `-j2`, sm_121, 718 s. | `llama-server --version` reports build 11514 / `de7fa0a3c`. `llama-cli --list-devices` sees CUDA0, the NVIDIA GB10. | [`llamacpp-b11514-de7fa0a3.json`](../../../configs/build-manifests/llamacpp-b11514-de7fa0a3.json) |
| exllamav3 + TabbyAPI | exllamav3 v1.6.1 `f45b8451`, no patches; it builds unpatched on aarch64. TabbyAPI `2636ef60`. | Wheel built against a copy of GLM-5.3's `native-runtime-003` (torch 2.13.0+cu130). `MAX_JOBS=2`, sm_121a, 738 s. vllm-exl3 removed, TabbyAPI's three missing dependencies added. | `exllamav3_ext` loads. `MiMoV2ForCausalLM` registers. A 1024² GPU matmul runs on the GB10 (44 MiB peak). TabbyAPI's exllamav3 backend imports and `main.py --help` runs. `uv pip check` reports 209 packages compatible. | [`mimo-v26-flash-exl3-runtime-001.json`](../../../configs/build-manifests/mimo-v26-flash-exl3-runtime-001.json) |

TabbyAPI's disk-cache options need exllamav3 1.6.2, which is unreleased, so they report unsupported. Both builds ran in a swapless user scope (`MemoryMax=64G`, `MemorySwapMax=0`) with production stopped and 115.61 GiB available. The runtimes live at `~/.cache/llamacpp-b11514-de7fa0a3/` and `~/.cache/mimo-v26-flash/native-runtime-001/`.

## The restore failed twice, then succeeded after the owner dropped the kernel caches

Production (`qwen38fn`, candidate 10) was stopped from **14:11:57Z to 18:40:21Z**, about 4 h 28 min. The builds took 25 min of that. The rest was a restore that the engine's own KV-arena guard refused twice, as it is designed to, plus waiting for a root-only cache drop.

All times below are UTC. The server log uses local time (UTC−4).

| Time | Event | Evidence |
|---|---|---|
| 14:11:54 | `52_engine_switch.sh stop`, rc 0. | `build-window/window.log`, `stop.log` |
| 14:11:57–14:36:39 | Both builds; see above. | `build-window/`, `exllamav3/` |
| 14:36:39 | `52_engine_switch.sh qwen38fn`. The 114 GiB admission passed with 114.09 GiB. MemFree was about 83.5 GiB and Cached about 23.9 GiB, almost all of it from the builds. | `build-window/meminfo.log`, `restore-switch.log` |
| 14:46:17–14:47:44 | 113 kernel lines `NVRM: nvCheckOkFailedNoLog: … Out of memory [NV_ERR_NO_MEMORY] … _memdescAllocInternal` while weights loaded. There were no Xid, OOM-kill or other kernel lines. | `build-window/kernel.log` |
| 14:48:10 | The worker released Flash-Next's own cache roots, then refused the KV arena: MemFree 14.63 GiB, arena 17.90 GiB plus 2.00 GiB margin. | `server-arena-lines.log` |
| 15:07–15:17:55 | The switch's rollback restarted the same profile. It started with MemFree 99.15 GiB and was refused again at 14.67 GiB. | same |
| 15:37:15 | Switch rc 1, `ROLLBACK VERIFICATION FAILED`. The host was idle at 114.88 GiB available. | `restore-switch.log` |
| 18:28:47 | The owner ran `sudo sysctl -w vm.drop_caches=3`. MemFree rose to 116.63 GiB, Buffers to 0.01 GiB and SReclaimable to 0.10 GiB. | `restore/window.log` |
| 18:28:57–18:40:21 | `52_engine_switch.sh qwen38fn`, rc 0. The arena check passed with MemFree 30.94 GiB. Low point 11.54 GiB; SwapFree unchanged during the load; the kernel capture is empty. The live process runs runtime-002's interpreter, with :8010 and :8013 listening. | `restore/`, `server-arena-lines.log` |

**Cause.** The NVIDIA driver does not reclaim page cache for its allocations. So candidate 9 added the worker's free-memory check, which releases page cache only under `QWEN38FN_HOST_RELEASE_ROOTS` and the model directory. Most of the roughly 24 GiB of page cache present at switch start came from the builds, which sit outside those roots. Block buffers and reclaimable slab are outside them as well. In three loads, MemFree at the arena ended up 82.6 to 85.6 GiB below MemFree at switch start:

| Load | MemFree at start | MemFree at arena |
|---|---|---|
| Promotion, 2026-10-12 | 105.08 | 22.44 |
| Failed rollback | 99.15 | 14.67 |
| Restore | 116.55 | 30.94 |

The first failed attempt is the exception (83.5 to 14.63), presumably because the kernel reclaimed some cache when MemFree reached its watermark. **A qwen38fn start needs about 105.5 GiB MemFree, not just 114 GiB MemAvailable.** The switch's admission measures only MemAvailable, so it admitted every one of these starts.

**What I did between the failures, which made things worse first:**
- `release_tree` (POSIX_FADV_DONTNEED as the serving user) on the build trees, `/usr/local/cuda-13.0`, uv's caches, the worktree and Flash-Next's roots took Cached from 14.43 to 2.97 GiB.
- A filesystem-wide `find` that I ran while looking for the server log had raised Buffers to 4.93 GiB and SReclaimable to 5.83 GiB. No user-level call releases either. MemFree reached only 102.35 GiB, below the 105.5 GiB needed.
- I then tried to force reclaim by streaming 96 GiB of model files through page cache (`build-window/cache-pressure.py`). It hit its safety stop: when MemFree reached about 4 GiB, the kernel swapped 0.33 GiB of idle anonymous memory instead of dropping the buffers and slab, which grew. The script then released everything it had read. Its terminal output, copied as printed:

```
11:43:03 start MemFree=100.47 MemAvailable=114.06 Cached=2.97 Buffers=5.73 SReclaimable=6.78 SwapFree=15.76
11:44:37 read 90 GiB MemFree=8.95 MemAvailable=114.00 Cached=92.98 Buffers=6.27 SReclaimable=7.72 SwapFree=15.76
11:44:43 SAFETY STOP MemFree=3.95 MemAvailable=115.27 Cached=98.39 Buffers=6.29 SReclaimable=7.72 SwapFree=15.43
11:44:44 end MemFree=99.77 MemAvailable=115.24 Cached=2.71 Buffers=6.29 SReclaimable=7.55 SwapFree=15.43
```

The switch has no delegated cache drop, so the restore waited for the owner's `drop_caches`. A watcher (`restore/restore-after-drop.sh`) ran the switch as soon as MemFree was at least 108 GiB, Buffers below 2 GiB and SReclaimable below 3 GiB.

## Follow-ups

1. **A switch-level cache release before every engine load.** The c10 record already listed this follow-up; today showed it is needed. The switch already runs as root, so it can drop caches with `sync; echo 3 > /proc/sys/vm/drop_caches`. Then it should admit on MemFree as well as MemAvailable: for qwen38fn, start MemFree ≥ arena + margin + about 86 GiB of load. A start like 14:36 would then neither log NVRM lines nor fail.
2. **The `14_build_llamacpp_pin.sh` memory gate.** This is the medium finding from #16's review. Its contained-build path permits a build with MemAvailable as low as 11 GiB under an 11 GiB cgroup, which can take a production host to zero. `12_build_qwen38_llamacpp.sh` and `13_build_laguna_llamacpp.sh` have the same gate.
3. **Don't run filesystem-wide scans** (`find /`) while an engine load is pending. Their buffers and slab count as available memory but sit outside every release root.

## Files

- **`build-window/`**: `window.log` (the orchestration), `stop.log`, `restore-switch.log`, `kernel.log`, 1 Hz `meminfo.log`, the llama.cpp build, verification and `--help` output, the exllamav3 step log and `cache-pressure.py`.
- **`exllamav3/`**: the wheel build, install and uninstall logs, `pip check`, the import/GPU/TabbyAPI verification, and timing.
- **`restore/`**: the successful restore's `window.log`, `switch.log`, empty `kernel.log`, `meminfo.log` and the watcher script.
- **`server-arena-lines.log`**: the start, KV-arena and refusal lines of the three loads, from the switch's `qwen38fn.server.log`. The full argument lines are elided; the profile render is in the repository.
