# GLM 5.3 model-free probe evidence

This protocol covers the synthetic native checks and the cache allocator probe.
It does not establish model fidelity, processed context, serving performance, or
permission to launch a production profile. The Python identity review remains
paused; implementing this additional evidence component does not resume it.

An authoritative probe verdict requires **all** of the following:

1. The frozen source, scorer, configuration, complete runtime inventory,
   metadata/tokenizer/fixture inventories and selected interpreter verify.
2. A BLS-verified public beacon was published after the freeze, and the executed
   seed and fixture order follow the frozen derivation.
3. The frozen inner probe passes its complete acceptance contract.
4. `score_host_observations` in `scripts/lib/glm53_host_evidence.py` passes against
   the caller's frozen expected arguments and directly captured host records.
5. The complete runtime and frozen input files still verify after execution.

Missing evidence or a failed check is not a successful attempt. Preserve the
attempt and its failing assertion. The host scorer's own `PASS` has the scope
`host_and_probe_identity_observations_only`; it cannot substitute for this
combined verdict or any subsequent model qualification gate.

## Capture contract

The controller writes the following files into a fresh attempt directory. It
must capture observations directly; the unit fixtures in the test suite are
synthetic mutations and must never be used as qualification evidence.

| Files | Producer and binding |
| --- | --- |
| `manifest.json`, frozen code and inventories | Controller freezes exact source/configuration/interpreter/runtime/metadata/scorer hashes before requesting randomness. |
| `randomness.json`, relay receipts, verifier output | Two agreeing public relay responses and the pinned BLS verifier; publication must follow the freeze. |
| `invocation.json` | Actual wrapper argv, cleared/selected environment, source freeze digest, beacon digest and start timestamp. |
| `wrapper.log` | Direct stdout/stderr of the existing GLM cgroup wrapper, with `GLM_SAFE_DONE_DIGESTS=1`. Its terminal receipt binds `main.log`, `samples.log` and `kernel.log`. |
| `main.log`, `samples.log`, `kernel.log` | Exact bytes copied from the crash directory named by that wrapper receipt. No filtering or repair. |
| `identity/manifest.json`, `identity/raw.jsonl`, `identity/summary.json` | The frozen, separately reviewed Python probe guard. The host scorer binds its executable, source files, argv, seed arguments and environment to the caller's expected values. |
| `checks/manifest.json`, `checks/raw.jsonl`, `checks/summary.json` and probe logs | The inner native or cache probe; retain every required assertion and failure. |
| `unit-live.json`, `unit-after.json` | Actual `systemctl --user show` command, return code, stdout, stderr and capture timestamp; use the exact argv returned by `unit_query(unit)`. |
| `cgroup-after.json` | Direct post-stop observation of the exact `/sys/fs/cgroup` path. Only `ENOENT` means absent; permission or other observation errors fail. |
| `swap-before.json`, `swap-after.json` | Unmodified `/proc/vmstat` text, source path and capture timestamp, covering the interval from before wrapper start through cgroup cleanup. |
| `summary.json` | Fixed combined verdict, unrounded observations, hashes and any failing assertion. |

The unit records have keys `command`, `returncode`, `stdout`, `stderr` and
`observed_at`. Capture the live record while the matching fresh unit is running;
record its actual memory limits, swap limit, OOM policy and kill mode. Observe
the same unit after exit. A stale unit, surviving MainPID, remaining cgroup, failed
query or missing live observation fails the attempt.

The cgroup record has keys `path`, `exists` and `observed_at`. The swap records
have keys `path`, `text` and `observed_at`. Timestamps are UTC Unix seconds read
at capture. Do not infer unit properties from the requested launch parameters or
write expected values into an observation record.

## Contained invocation

Retain the existing global inference lock and hardened wrappers:

```text
results/glm52-gates/harness/glm_cgroup_run.sh
results/glm52-gates/harness/glm_safe_run.sh
```

The model-free probe requires stable start memory of at least 110 GiB, a 40 GiB
whole-host kill floor, fresh cgroup containment, zero cgroup swap, a bounded
timeout and digest-bearing completion. Let the existing wrapper derive its
limits from the actual preflight; do not copy an old expert-arena ceiling.

Inside that containment, run the final packaged interpreter with `-I -B`, the
frozen identity guard, and then the frozen probe. Pass only the declared
environment through `env -i`. Keep JIT/cache paths outside the immutable runtime
prefix, and disable network metadata fallback. The selected diagnostic flags,
including the temporary BF16 geometry candidate flag where required, belong in
the frozen environment and its digest.

The identity guard must receive both independent review closure and a new freeze
before this invocation can become authoritative. Its local proposed correction
has not received that closure. Previous native attempts 001–003 therefore remain
`NO_RESULT` for continuous Python identity despite attempt 003's fourteen passing
inner kernel assertions.

## Host scorer inputs and limits

Freeze these expected arguments alongside the scorer:

```text
binary_sha256, executable, guard, guard_sha256, probe, probe_sha256,
probe_arguments, environment_sha256, unit_prefix,
kill_floor_gib, minimum_start_gib, timeout_seconds,
maximum_sample_gap_seconds
```

`probe_arguments` includes the actual public seed. The expected environment digest
uses the guard's exact sorted NUL-separated `NAME=value` byte representation.
The guard/probe paths identify the copied frozen files, not mutable repository
paths. The unit prefix identifies one attempt; retries require another attempt
directory and prefix.

The current scorer is intentionally limited to this Spark's benchmark-owner
user cgroup. A production dsv4 service or another host requires its own lifecycle
and containment qualification. It checks at least three external memory samples,
periodic and final Python identity, at most two seconds between samples, matching
sampling windows, zero new whole-system swap counters, clean cgroup event counters,
and actual unit/cgroup teardown. It requires exactly one wrapper start, end and
completion record, validates the frozen tag, memory floors and timeout, and
rejects contradictory or malformed terminal records. Wrapper control/process/final
and end timestamps must agree with the identity, external sampling and cleanup
windows. Process discovery may follow the first identity observation by the
frozen sample gap, because the wrapper waits briefly before discovering its child. Its minimum available-memory result comes from
the raw external samples; allocator self-reports do not supply that metric.

The controller must reject output-recording errors and unexpected wrapper exit,
retain ownership of the inference lock through descendant cleanup, and verify
the output/runtime hashes before computing the combined verdict. No hardware
qualification or production switch follows merely from a passing CPU test suite.
