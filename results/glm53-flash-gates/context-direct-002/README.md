# Direct context attempt interrupted: NO_RESULT

Four 250,128-token requests were admitted and partial prefill progressed, but
none completed before a verified test-harness memory defect was found. The
client scored metrics by retaining the complete log in memory. The log already
exceeded 300 MB; a bounded reproduction and focused reviewer assessment showed
that continued growth could breach the host floor at end-of-run scoring.

The identity-verified client was terminated while GLM remained available. This
is not a full-context result or a claim that the model failed inference. The
archive preserves all input, raw stream and metrics bytes, cancellation, and
the bounded reproduction. No rows were filtered or replaced.

The new regression test gives a genuine RED on unchanged scoring code:
a 16 MB synthetic metrics log adds 30,189,420 traced peak bytes, exceeding the
fixed 5 MiB growth limit. The smallest proposed correction streams the same
raw log through EOF, keeping scalar checks and all existing acceptance rules.
The model/runtime and already reviewed components need no changes.
