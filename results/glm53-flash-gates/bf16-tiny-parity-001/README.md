# Tiny native loading/parity preparation 001: FAIL

The CPU-only synthetic probe reached stock checkpoint loading, then its evidence
writer rejected Transformers loading-info sets as non-JSON data. No tensor or
forward-parity verdict was reached, and no real model weights were loaded.
The complete original source, stdout, stderr, partial JSON, synthetic checkpoint
and failure records are retained. This is a preparation failure, not native GLM
reference data, fidelity, context, GPU memory or speed evidence.

The bounded repair must only serialize native loading-info sets deterministically
as sorted arrays. A local serialization check must reject unsupported objects;
the same tensor/dtype equality, per-stage meta residency and three sequence-length
assertions then run unchanged in a fresh directory. Do not replace this attempt.
