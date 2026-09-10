# Native direct context006: FAIL

All four requests processed250128 actual input tokens (1000512 aggregate), with
matching prompt IDs/usage and four overlapping output streams; no preemption.
The unchanged scorer fails because slots0,2,3 exhausted2048 tokens in reasoning
and returned empty final content. Slot1 passed retrieval and its negative control
with1584 total output tokens. Reasoning is not substituted for a final answer.

The actual named-profile lifecycle separately passed: native authentication,
READY, stop, terminal identity guard, no surviving model processes, and memory
recovery. All2463 prepared cache files remained unchanged after normalized
run-root relocation; only usage metadata and an empty lock were added. Closed
runtime/model verification passed after shutdown. No Xid/OOM/cgroup swap.
Minimum available memory was18.293609619140625GiB in15622 external samples.
Qwen recorded/default state and proxy/guard identities remained unchanged.

The unchanged scorer reproduced exactly and its17 existing tests passed. Source,
fixture/prelaunch/final bindings, public BLS randomness, requests, token streams,
full metrics, cache bytes, raw lifecycle/host/identity/terminal observations and
both review attestations are retained in the split archive. Concatenate parts
in numeric order to reconstruct attempt.tar.gz; verify manifest hashes before
extracting. The API key is deliberately excluded; all retained bytes were
scanned for it. No model weights or entire runtime payload is duplicated.

This is a context FAIL, not full qualification or qualified performance.
The next bounded alternative is committed separately in
[context-clear-instruction-001](../context-clear-instruction-001/PROTOCOL.md).
