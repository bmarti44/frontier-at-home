# Direct aggregate context 007: PASS

The native `glm-5.3-flash/cuda-spark-128g-1m-experimental` profile processed
250,128 actual input tokens in each of four concurrent requests: 1,000,512 total.
All four final answers retrieved the three expected records in order and passed
the negative controls. All finished normally. Actual prompt IDs, usage and
timestamped output streams matched; the fixed scorer verified simultaneous
generation and no preemption.

The configured capacity is 1,048,576 aggregate tokens in four 262,144-token slots.
This result covers that aggregate configuration; each individual request remains
capped at 262,144 tokens. It is one declared confirmation, with no statistical
fidelity or performance claim.

The [candidate protocol](../context-clear-instruction-001/PROTOCOL.md) clarified
only the test's output instruction after the preserved
[context 006 failure](../context-direct-006/README.md). Weights, model settings,
runtime and retrieval scorer stayed unchanged. Source revision `e008b707` was
frozen before public drand round 6452843; two relay responses and independent BLS
verification are retained. Inputs were prepared before loading the model. A
4,224-token startup correctness falsifier passed before the direct requests.
No causal fix for the older asynchronous CUDA failure is claimed.

Native named-profile start, status, authentication, READY, stop and cleanup
passed. The external watchdog retained 15,504 samples, with a minimum of
18.44664764404297 GiB available memory and zero cgroup swap. No Xid/OOM was
recorded. The identity guard retained 17,216 samples, exited zero and confirmed
no surviving process group. Memory recovered above 110 GiB; recorded default,
proxy listener and guards stayed unchanged. Full runtime/model verification
passed again after shutdown.

All 2,463 prepared cache files remained unchanged after the declared run-root
path relocation. The launch-versus-post-run comparison found growth only in
unfrozen usage metadata, from 1,498 to 2,072 bytes; its final bytes are retained.
No newly compiled kernel is used to support this confirmation.

The unchanged scorer reproduced the saved API summary exactly, and all 17
existing scorer tests passed. Both persistent reviewers independently rescored
the result and found no high/critical issue in this scoped evidence. Their exact
attestations are archived under `observations/`.

The [manifest](manifest.json) binds every archive file and split part. Concatenate
`attempt.tar.gz.part-*` in numeric order, verify its SHA-256, then extract. The
archive contains the original client streams and metrics, provenance-preserving
combined `raw.jsonl`, short correctness check, all input versions, frozen source,
public randomness, prepared cache bytes, host/identity/lifecycle observations,
post-run verification and reviews. Original prelaunch and final launch-bound
manifests are both retained. The API key is excluded, and every retained byte
was scanned for it. Model weights and the full runtime payload are bound by
closed inventories rather than duplicated here. Every archived file was read
back and checked against its size and hash.

The original API-only `client/summary.json` keeps its generic pending-host label
unchanged. The public [combined summary](summary.json) adds the completed
host/freeze/lifecycle checks. Full model qualification remains pending: the
100-case paired fidelity gate, production switching and qualified production
performance are not established by this context PASS.
