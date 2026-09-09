# Native top-k bounded diagnostic

Run the installed selection operator with 128 rows, 512 selected pools and the
failed batch's four request-length geometries. Finite monotonically increasing
scores are the control; all-NaN scores are a robustness diagnostic. No model is
loaded and returned indices are never used for a memory gather.

The fixed checks are: intact output canaries; every selected index within its
row's valid candidate range; no duplicate selections; and, for finite inputs,
exact agreement with an independent set of the 512 largest candidate positions.
Raw input/output tensors and operator completion/error observations are retained.
An invalid NaN result is not proof that model logits were nonfinite during the
long-context failure. Crash attribution remains NO_RESULT.

Use separate fresh contained processes, the inference lock, a 40 GiB host kill
floor, a 180-second wall timeout and identity monitoring. Stop the serving model
and require at least 110 GiB available before either process starts. The two
small arms use persistent pinned input staging and the installed native binary;
no CUDA build or runtime modification is authorized by this diagnostic.

## Observed result

Both fresh contained cases completed with wrapper exit0, identity checks passed,
zero cgroup swap and intact output guards. All128 rows in both cases returned
512 distinct indices within the valid candidate range. The finite case matched
the independent expected selection exactly. The all-NaN input did not reproduce
the comparison-only concern from CPU triage. This null result does not prove
all nonfinite patterns safe or explain the model crash. Full raw tensors,
identity observations, memory samples and binary/source bindings are archived.
No source or serving configuration was changed.
