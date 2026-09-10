# Native named-profile direct context attempt after startup corrections

Use glm-5.3-flash/cuda-spark-128g-1m-experimental through the actual profile
lifecycle. It has four 262,144-token slots (1,048,576 aggregate), native async
execution (CUDA_LAUNCH_BLOCKING absent), batch128/threshold32 and the existing
9,565,304,320-byte cache reservation. Keep the existing authenticated endpoint,
Qwen recorded/default state, shared inference lock and exact containment.

This is a new configuration, not a one-option replay of004: it removes CUDA
synchronization, includes profile readiness history, uses a prepared decode
kernel cache and selects fresh fixture values with a new public seed. Freeze
source, profile, runtime/model inventories, prepared cache and this request
preparation program before obtaining verifiable public randomness. A run-root
path is an ephemeral profile substitution; retain the exact resolved argv/env
and relocated cache hashes as launch observations. No model settings may vary.

The committed prepare_inputs.py uses the unchanged48 fixture builder and changes
only max_tokens from256 to2048 before writing final request hashes. Each input
remains250128 actual tokens; input plus maximum output is252176, below262144.
It generates all four fixtures from the same new public seed using the existing
per-slot derivation. Retain the seed receipt and all final input files.

The existing48 scorer stays byte-identical: all four streams must finish with
stop and valid final retrieval/negative-control answers, observed input/output
IDs and usage must agree, generated streams must overlap, four running requests
must be corroborated, and no preemption is accepted. Longer reasoning alone is
not a PASS. Do not change prompts, expected records or scorer to rescue a result.

Observe the actual profile's native authenticated readiness request and the
independent lifecycle check from profile-launch-001. Record both in startup
history before the four context requests. There is no causal attribution to a
single difference from003/004. Preserve every failed or null outcome.

Use the existing 18 GiB kill floor, 92/94 GiB cgroup, zero cgroup swap and9000s
server wall timeout. Record continuous identity, memory, kernel journal and
terminal cleanup. At least1,000,000 actual input tokens must be processed;
configuration or admitted tokens alone do not pass. New/mutated compiled inputs,
missing coverage, OOM, Xid, swap, timeout or surviving descendants invalidate
confirmation. Startup vLLM usage metadata is retained separately from compiled
cache bytes and is not presented as an immutable kernel artifact.

This attempt cannot establish paired fidelity, production switching or qualified
performance. Those gates remain separate, and GLM stays optional and estimated.

Prepare every long fixture while the model is unloaded. The frozen
fixture-profile/launch.json contains declared arguments solely for fixture
construction; it is not a server observation. After actual profile startup, the
frozen bind_launch.py first verifies every prepared file/source and requires
argument equality, preserves the prelaunch manifest/configuration, and binds
the actual launch.json. It changes no request, token ID, fixture or expected
answer. The actual resolved environment and relocated cache must also match
the frozen template before sending any long request.

This candidate also includes the reviewed orderly API shutdown sequencing,
--shutdown-timeout10, and the captured native cooperative autotune cache.
The prior profile's successful start/auth/stop cleanup observations remain, but
its guard-handshake failure and CPU-preparation timeout are not relabeled.
