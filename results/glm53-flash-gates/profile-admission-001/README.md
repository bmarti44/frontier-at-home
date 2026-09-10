# Named-profile admission on an occupied host

The fixed runner calls the actual named GLM start while server019 owns the GPU.
Acceptance requires explicit 110 GiB memory rejection before creating launch
artifacts or a unit, unchanged live model process identities/service invocation,
and intact guard services. This is a negative admission test, not a model load,
context result, performance result or complete lifecycle qualification.

The runner and acceptance formula are committed before execution. No inference
request is sent, and the current context replay's inputs/settings stay unchanged.

The actual host check passed. The named start exited with the expected 110 GiB
admission error, created no launch directory/unit, and preserved all three
existing model process identities and the systemd invocation. Both guard
services stayed active. Raw before/after observations and exact CLI output are
retained. The current context requests were not touched.
