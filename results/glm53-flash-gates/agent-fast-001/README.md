# Agent preset: basic serving PASS

The optional 64K-per-request, four-slot preset started and passed authenticated
chat, rejection without authentication, correct tool arguments, a tool-result
round trip, four overlapping requests, four images in order and a 16-frame video.
Media fixtures were 224x224. No external weather service was called.

The same model weights are used with a 4 GiB KV reservation and 512-token prompt
batches. The full 64K context and model fidelity have not been qualified. The
original million-token experiment remains available and its failure is preserved.
Qwen remains the recorded default. This is a live snapshot, not a terminal result.

The archive contains raw stream timestamps, requests/responses, launch settings,
identity observations, memory samples and scorers. It excludes the private API
key and warm caches. Short development timing measurements are preserved inside
`agent-timing/`; the second case overlapped a media request and is excluded from
isolated speed comparisons. Both coding responses reached their requested token
limit, so they do not establish completed coding-task quality. Qualified
production performance remains not yet measured.

## Terminal update

The session received SIGTERM at 20:55 EDT on September 9. The wrapper reported
interruption and the guard reported BrokenPipeError during shutdown. The sender
is unknown. Terminal FAIL is preserved in `terminal/`; all model processes were
absent afterward and memory recovered. The earlier functional snapshot is intact.
