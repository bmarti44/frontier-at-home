# Lever screen attempts (DIAGNOSTIC)

## Window s1, 2026-11-23: a3 refused at preflight by its own driver's lock race

**What happened.** Window s1 ran `a0 a3 a4 a5 a1 a2 a0r`. a0 completed and exited 0 at 04:53:50Z. a3 started in the same second, and its preflight `flock -n` on the inference lock failed:

```
2026-11-23T04:53:50Z arm a3 start
/run/lock/frontier-at-home/inference.lock is held: an engine is serving; stop it first
2026-11-23T04:53:51Z arm a3 exit 2
```

a4 started one second later and took the lock normally.

**Cause.**
- `run-arm.sh` takes the lock on fd 8 before it starts its 1 Hz meminfo sampler and its kernel-journal follower in the background. Both inherited fd 8.
- At exit, `cleanup` kills the sampler subshell, but the subshell's in-flight `sleep 1` child survives the kill and holds fd 8 for up to one more second.
- `window.sh` starts the next arm at once, so a fast preflight can meet the lock still held by the previous arm's leftover `sleep`.
- `run-effort.sh` shares the same preamble.

It recurred when a5 exited at 05:15:29Z. a1, a2 and a0r each started and were refused within 05:15:30Z, so window s1 ended after three arms: a0, a4 and a5, all complete. The window then restored production.

**Reproduced with a stub** (driver takes fd 8, backgrounds a `while sleep` subshell, kills it 0.5 s later, exits):

| Shape | Lock right after the driver exits |
|---|---|
| s1 shape | held |
| subshell started with `8>&-` | free at once |
| s1 shape, next driver `flock -w 15` | acquired after 1,501 ms |

**Consequence.**
- Every refusal was in the preflight, before `mkdir "$OBS"`, so a3, a1, a2 and a0r have no directory, and no evidence was lost. They are still due (`--may-run`).
- No engine was started or left behind.

**Fix,** applied after window s1 had closed (bash reads a running script incrementally), to `run-arm.sh` and `run-effort.sh`:
- The sampler and the journal follower start with fd 8 closed (`8>&-`), so only the driver shell holds the lock.
- The preflight waits up to 15 s for the lock (`flock -w 15`) instead of failing at once. A serving engine holds the lock indefinitely, so this still refuses while production is up.

**Next.** Window s2 runs `a3 a1 a2 a0r`. a0r still runs last of all lever arms, so a0 (first in s1) and a0r (last in s2) bracket every lever arm, as PREREG's baseline pair requires. No extra repeat is needed.

**a6 authored before window s2** (`cuda-spark-128g-1m-tune-dspark-humming`, `"levers": ["dspark7", "moe_humming"]`, `min_concurrency` 3.0). Against a0 alone, window s1 keeps DSpark (decode geomean 1.099, code 1.352) and humming (rp 1.237, probes byte-identical). Authoring a6 now lets s2 run it right after a0r, without another production stop.
- `--may-run` admits a6 only when a1–a5 all exist and the scorer's kept set, with a0r and a3/a1/a2 counted, is exactly these two levers.
- If the kept set changes, a6 is refused without a directory, and a corrected combined arm (a6b, the last combined entry, which the scorer reads) is authored before it runs.
- Window s2's `effort:<arm>` entries are admitted by `--effort-due` only for the carrier of a `COMPLETE` screen. The others refuse in their preflight.
