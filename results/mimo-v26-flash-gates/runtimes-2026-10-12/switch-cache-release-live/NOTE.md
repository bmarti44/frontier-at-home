# Switch cache release and MemFree gate: first live run (clean)

This is follow-up 1 of [the runtime-build record](../RESULT.md). PR #22 (`a4044035`) makes every engine start drop the kernel caches and admit on MemFree. On 2026-10-12, after the merge, the owner approved one controlled `qwen38fn` stop and restart through the new path. The orchestration script is `live-check.sh`.

**Steps** (UTC, `window.log`):
1. The production checkout was fast-forwarded to `a4044035` and `verification/MANIFEST.sha256` checked out.
2. At 22:41:00, `sudo -n scripts/52_engine_switch.sh stop` returned rc 0. MemFree went from 8.18 to 111.81 GiB, with Cached at 4.33 GiB.
3. At 22:41:08, `sudo -n scripts/52_engine_switch.sh qwen38fn` ran and returned rc 0 at 22:51:34, about 10.5 min after the stop.

**What the new path did** (`switch.log`, `kernel.log`):
- **Cache release:** `host caches released: MemFree 116.39 GiB, MemAvailable 115.84 GiB`. The kernel logged `bash (…): drop_caches: 3` at 22:41:09.
- **MemAvailable guard:** passed at 115.84 GiB against the 114 GiB required, with three stable samples.
- **New gate:** `free-memory gate: MemFree 116.39 GiB >= 114 GiB`.
- **KV arena:** the worker found MemFree 30.86 GiB against the arena's 17.90 GiB plus 2.00 GiB margin (`server-arena-lines.log`). This matches the 2026-10-12 restore after the owner's manual drop, which found 30.94 GiB.

**Checks:**
- **Kernel:** the only kernel line in the window is the `drop_caches` notice. There were no NVRM, Xid or OOM lines.
- **Swap:** SwapFree went from 16,486,556 to 16,486,644 kB, so it did not grow.
- **Memory:** idle after the load, MemAvailable was 11.50 GiB, the same as before the stop.
- **Process:** the live engine runs runtime-002's interpreter (`live-process.txt`), with :8010 and :8013 listening.
- **Serving:** the switch's own authenticated validation passed (rc 0).

**Files:**
- `window.log`
- `stop.log`
- `switch.log`
- `kernel.log`
- 1 Hz `meminfo.log`
- `server-arena-lines.log`
- `live-process.txt`
- `live-check.sh`
