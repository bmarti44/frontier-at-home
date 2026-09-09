# GLM is serving text, tools, images and video: basic smoke PASS

The authenticated local API returned `2 + 2 equals 4.` and rejected an
unauthenticated model-list request with HTTP 401. A tool request produced
`get_weather` with `{"city":"Paris"}`. Four overlapping client requests all
completed with correct arithmetic answers. No external weather tool was invoked.

This working launch enables text/tools/images/video, 128-token prefill chunks, the standard
allocator, one-time warm-up cache cleanup, prepared attention/sampling libraries
and native attention tactics without autotuning. It retains four 262,144-token
slots and the explicit full cache. These are short serving checks; maximum
context and fidelity are not qualified. The separate context-direct-003 run failed
with a CUDA illegal access and Xid31; this restore does not resolve that fault. Single-image, four-image and16-frame video checks passed with224x224 fixtures. Larger media remain unqualified.

The archive is a consistent byte snapshot of individual files while the server
continues running; each file has its own hash. It includes requests, responses,
launcher code/configuration, identity observations and external memory samples.
It excludes the API key and reproducible warm caches. The terminal lifecycle
and clean shutdown have not been observed for this still-running attempt.
Qwen remains the recorded default. The test server has a 2.5-hour timeout.

See [the one-command launch guide](../../../docs/GLM53-QUICKSTART.md).

## Terminal update

Session017 was stopped with identity-verified SIGINT for a bounded native
operator test. The wrapper completed with exit0 and no kill. Terminal raw
samples and the shutdown summary are in `terminal/`. This does not qualify
production switching or full context.
