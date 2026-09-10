# Second bounded startup reclamation experiment

The committed `soak-native-008` attempt passed short correctness and reused all
2,486 prepared cache files, but failed the whole-host gate: one swap-out page and
4 KiB used-swap growth. GLM cgroup swap remained zero. The same 40 MiB CUDA
reservation decrease as attempt 007 did not reliably close this finding. A
simultaneous snapd VmSwap increase is correlation, not causal attribution or an
exemption. The failed result and both publication reviews remain immutable.

This is the second alternative already specified by
`startup-final-warmup-001/PROTOCOL.md`: call process-local libc `malloc_trim(0)`
after existing startup garbage collection, before and after the parent's memory
profiling method. Select it only with `GLM53_TRIM_STARTUP_HEAP=1`, resolved once
at module initialization. Bind libc and define observation helpers only when
enabled. Disabled mode must retain the existing worker method and perform no
heap-trim calls, observation I/O or logging. Preserve the parent return, live
objects and failure propagation. The production-path CPU test must first fail
on unchanged code because unused CPU heap pages remain held before profiling.
Commit that genuine RED before implementation.

Record actual process RSS and anonymous RSS before and after each trim, its
return value, PID, phase and wall-clock timestamps. These startup observations
are diagnostic only; external timestamped whole-host memory/swap, cgroup and
identity samples remain authoritative. Zero release is a valid null result.
No token/layer/expert path changes, global VM settings or new privileges.

Enable the new flag only in the experimental profile and remove its
`GLM53_RELEASE_WARMUP_CACHE` selection. Keep the prior load/file-cache flags,
512/128 scheduling, four 262,144-token slots, aggregate cap 1,048,576, weights,
tokenizer, precision, fixed KV budget, media policy, containment, host floor,
timeouts and inference lock unchanged. Agent and production profiles, default,
closed scorers and previously reviewed components remain byte-identical.

After clean source and scoped review, freeze source/runtime/model/cache/scorer/
fixtures/configuration, then obtain a later verified public seed. Admission
also binds the actual `malloc_trim` provider resolved with `dladdr` under the
pinned runtime Python, without invoking the trim. Freeze that library's path and
hash; verify its mapped device/inode/path in the identity-verified workers after
launch and retain those observations. This is startup evidence only. Admission
requires the unchanged host checks, authenticated short correctness, unchanged
prepared kernels and fixed necessary window before the full-duration client.
Any host swap I/O, used-swap growth, memory-floor violation, cgroup failure,
missing identity, malformed evidence, wrong output or failed shutdown retains
FAIL. New kernel files retain NO_RESULT for frozen-cache confirmation. No
shorter test is a million-token result and no diagnostic timing is headline speed.

The experiment helps only if it releases actual anonymous RSS and the unchanged
host gate passes. A release alone does not prove causation or qualification.
If neither bounded reclamation alternative helps, close this reclamation branch
as NO_RESULT with its measured finding; do not start another speculative memory
variant. Direct current-configuration context, paired native fidelity, sustained
use, maximum media and production switching remain separate qualification gates.
