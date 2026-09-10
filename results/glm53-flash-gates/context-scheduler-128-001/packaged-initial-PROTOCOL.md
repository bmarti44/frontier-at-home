# Return the existing optional 1M profile to the 128/32 scheduler

This is a bounded declarative fallback, not a new engine or profile variant.
The completed indexer workspace preparation failed repeated-control ordered-byte
identity before either 64 MiB arm ran. Preserve that FAIL; do not normalize its
indices or continue that workspace route.

## Fixed implementation scope

Change only two operational values in
`configs/profiles/glm-5.3-flash/cuda-spark-128g-1m-experimental.json`:

- `--max-num-batched-tokens`: `512` to `128`.
- `--long-prefill-token-threshold`: `128` to `32`.

A factual `status.basis` explanation may be updated; `status.state` remains
`estimated`. Apart from those two values and that explanation, the profile must
remain byte-for-byte equivalent after JSON parsing. Preserve all current startup
reclamation switches. Do not change the native math, weights, precision, cache
format, indexer workspace budget, kernel sources, loaders, diagnostic selection,
serving lifecycle, authentication or closed evidence scorers.

Keep aggregate context exactly 1,048,576, four simultaneous slots of 262,144,
KV reservation 9,565,304,320 bytes, four images or one video with the existing
16-frame bounds, media sizing, tool/reasoning settings, localhost authenticated
endpoint and optional profile name. Keep minimum start memory110 GiB,
whole-host kill floor18 GiB, cgroup MemoryHigh92G/MemoryMax94G/MemorySwapMax0,
existing wall-clock limits, monitoring, identity checks and clean group shutdown.
The agent-fast profile, production profile, current default, proxy and reboot
configuration are unchanged. This does not promote GLM or make it the default.

## Test-first evidence

Before implementation, retain the complete original profile as
`profile-before.json` with its exact hash, source revision and the updated tests.
Run `scripts/tests/test_glm53_experimental_profiles.py` with the pinned packaged
Python `-I -B` against the unchanged 512/128 profile. The batch128 and explicit
long-prefill32 assertions must fail for their actual rendered values. Keep the
agent-fast512 expectation and every other existing assertion. Commit tests,
protocol and genuine RED before changing the profile. The test stage does not
start a model or service, build anything or execute GPU work.

After implementation, run the complete focused profile tests and verify the
parsed profile differs only in the two scheduled values and permitted status
text. Existing resolver, launcher47, profile entry93, auth, status and stop
implementations require no production changes for these values.

## Fresh confirmation and limits

Direct007 historically passed four250,128-token inputs (1,000,512 actual input
tokens) under the128/32 scheduler. Its result is evidence for that recorded
configuration only. Current startup flags, source bindings, host state and cache
inventory must be frozen anew and require a new verifiable public seed and
current-settings confirmation. Do not relabel direct007 as the new candidate.

For direct context, reuse the unchanged128/32 validator and scorer in
`scripts/48_probe_glm53_context.py`, the existing actual-launch binder, and the
current clear-instruction preparation/short correctness check. The frozen
512/128 adapter is not the selected validator for this fallback. A prospective
freeze recipe must bind every actually selected source and exact rendered
argv/environment, verified runtime/model inventory, tokenizer, prepared cache,
fixtures, request options and public randomness before execution. Existing
prepared128/32 kernels may be reused only after their normal hash/path checks;
cache changes and runtime bytecode mutations still fail their fixed checks.

A later direct confirmation requires four completed correct retrieval streams,
actual token counts/overlap and all unchanged host, cache, identity and terminal
lifecycle checks. No truncation, swap, floor breach, OOM, Xid, missing coverage,
short output or surviving descendant is acceptable. Run it serially with no
publication, build, reference probe or other auxiliary workload during startup
or inference. Preserve every negative attempt. This test-stage record contains
no new context, durability, fidelity or performance result; those gates remain
separate. Any subsequent full durability run must use its unchanged fixed
acceptance and a separate fresh server.
