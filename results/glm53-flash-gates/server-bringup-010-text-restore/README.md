# GLM is serving text and tools: basic smoke PASS

The authenticated local API returned `2 + 2 equals 4.` and rejected an
unauthenticated model-list request with HTTP 401. A tool request produced
`get_weather` with `{"city":"Paris"}`. Four overlapping client requests all
completed with correct arithmetic answers. No external weather tool was invoked.

This working launch uses text/tools mode, 128-token prefill chunks, the standard
allocator, one-time warm-up cache cleanup, prepared attention/sampling libraries
and native attention tactics without autotuning. It retains four 262,144-token
slots and the explicit full cache. These are short serving checks; maximum
context and fidelity are not qualified. Images/video remain disabled here.

The archive is a consistent byte snapshot of individual files while the server
continues running; each file has its own hash. It includes requests, responses,
launcher code/configuration, identity observations and external memory samples.
It excludes the API key and reproducible warm caches. The terminal lifecycle
and clean shutdown have not been observed for this still-running attempt.
Qwen remains the recorded default. The test server has a 2.5-hour timeout.

See [the one-command launch guide](../../../docs/GLM53-QUICKSTART.md).
