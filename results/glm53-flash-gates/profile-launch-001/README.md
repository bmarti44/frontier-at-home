# Actual profile start/stop: clean-shutdown failure

The real93 profile start command emitted ready, status reported running, and
independent native health/authentication/model/reply checks passed. The matching
stop command returned zero, status reported stopped, all model processes left,
and memory recovered above110GiB. Qwen state, proxy identity and guards stayed
unchanged. These observations establish executable profile wiring.

The fixed complete lifecycle verdict remains FAIL: whole-group termination
killed the guard parent before the API child completed its terminal handshake,
causing an uncaught BrokenPipe traceback. A semaphore warning is also retained.
Zero command exit status is not substituted for a successful handshake. The
next correction must stop the identity-verified API first and allow the guard
to finish; whole-group cleanup remains a bounded fallback, never a clean PASS.

No long-context request was sent. The separately contained CPU preparation
attempt hit its180-second limit (MemoryHigh384MiB, MemoryMax512MiB, zero swap).
Its failure and partial inputs remain in the archive. Startup also regenerated
the cooperative autotune cache, whose subtree was missing from preparation.
That cache and the fresh freeze/public-seed receipts are retained. This is not
a frozen-cache or full-context result.
