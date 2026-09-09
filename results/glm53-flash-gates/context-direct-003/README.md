# Direct context execution failed

All four 250,128-token requests were admitted. During prefill the engine raised
CUDA illegal memory access; the kernel journal recorded Xid31 and all four
HTTP 200 streams carried server-error payloads with code 500. No request completed. The fixed
scorer rejected the error rows; it did not manufacture a capability result.

The asynchronous error surfaced during a CPU synchronization in FLA KDA chunk
index preparation, which does not identify the originating kernel. Full raw
inputs, streams and unfiltered metrics remain preserved. Model/runtime and
scorer bindings are in context-freeze-003. The model processes exited; the host
recovered above110GiB without reboot or new privileges. Root-cause investigation
must use bounded falsifiers before another large context run.
