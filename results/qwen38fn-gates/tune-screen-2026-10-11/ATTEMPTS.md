# Screen execution record

## a1, attempt 1 (2026-10-12T00:40:20Z): `FAILED_ATTEMPT`, a harness defect (preserved in `a1/`)

**What happened.** runtime-002 imported and loaded cleanly:
- the V2 Model Runner was in use;
- `checkpoint_mapped PLE: QWEN38FN_PLE_WILLNEED=0`;
- `Qwen4Exp MTP draft vocabulary: full`.

Then the engine's own arena guard (`scripts/lib/qwen38fn_worker.py`, `HostCacheReleaseWorker`) refused the KV arena and startup failed cleanly:

```
host page cache released under /home/bmarti44/models (496 files); MemFree 19.85 GiB,
KV arena 17.90 GiB + margin 2.00 GiB: refusing to allocate the KV arena
```

- The kernel log has no lines.
- memwatch logged `ARMED`, then `FAIL_CLOSED reason=missing_invalid_or_unverifiable_armed_target` when the engine exited. There was no `BREACH` or `EMERGENCY_STOP`.
- Swap did not grow.
- The scope was empty at exit.
- The server log's only `NVRM` matches are the guard's own refusal text ("… short of free memory (NVRM)"). It is not a driver message.

**Diagnosis.** The guard releases page cache only under the model root. This screen is the first time two runtimes have run back to back, and a0 left runtime-001's libraries cached.

Resident page cache after the attempt, measured with `mincore` over each tree:

| Tree | Resident |
|---|---|
| `native-runtime-001` | 2.56 GiB (from a0) |
| `native-runtime-002` | 2.46 GiB (from this attempt) |
| `/home/bmarti44/models` | 0.00 GiB (released by the guard) |

MemFree at the same arena check, by run:

| Run | MemFree | Runtimes cached |
|---|---|---|
| candidate 9 | 22.75 GiB | one |
| a0 | 21.15 GiB | — |
| a1 | 19.85 GiB | two (needs 19.90) |

The shortfall equals the stale runtime-001 cache. It is not a property of runtime-002, and nothing unsafe happened: the guard failed closed, as it was built to.

**Fix (targets only this failure).**
- `run-arm.sh` now applies the existing `checkpoint_cache.release_tree` (POSIX_FADV_DONTNEED; mapped pages are kept) to every `~/.cache/qwen38-flash-next/native-runtime-*` tree. It does this in the preflight, before the memory guard, for every later arm.
- a1 is retried once as arm `a1r`, with the same profile, base and lever.

**How the scorer treats a1.** It voids a1 and moves on, but only if all of these hold:
1. its driver logged `FAILED_ATTEMPT`;
2. it has no `speed.json`;
3. it has no kernel or memwatch kill line;
4. its only server-log kill matches are the arena guard's refusal;
5. `arms.json` names exactly one retry arm with the same profile, base and lever, and that retry is not itself voidable.

Anything else stays scored as before (a1 is unsafe, so the screen is `NO_GO`). Changes made at the same time:
- the arena guard's refusal is no longer counted as a server-log kill line, and real `NVRM:` driver text still is;
- self-test cases cover a valid void, a void refused for a kernel NVRM line, a void refused for a completed arm, and a void without a retry arm.

**Acceptance test for the fix.**
- a1r's driver log shows the runtime trees released before the guard;
- a1r reaches ready with the arena check passing;
- the scorer voids a1, scores a1r under the unchanged L0 rule, and its self-test passes.

**Carried to c10.** Any start that follows a different runtime (c10's run 1 after production, and every arm change in the ABBA campaign) meets the same stale cache. For c10:
- `97_matched_speed_blocks.py` gains `--release-tree`;
- the promotion switch needs the switch-level page-cache release already listed as a follow-up.
