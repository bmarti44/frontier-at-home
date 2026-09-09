# KDA preparation 001 — FAIL

The probe failed before KDA cases with KeyError('linear_num_heads'). It tried
to read flattened KDA fields from raw checkpoint JSON; the checkpoint stores
these under linear_attn_config and the pinned Glm5NextTextConfig normalizes
them. This is an execution defect in the preparation harness.

The wrapper returned 1, and identity correctly rejected the missing successful
completion handshake. The failed attempt and all raw host/identity/output
records are preserved. No KDA analytic, binary, model, context or performance
result is claimed. No model weights were loaded. A generated CUDA helper is
archived as post-run preparation state, not as a qualified frozen binary.

The next candidate must use the pinned configuration normalizer, demonstrate
nested-config acceptance and changed-geometry rejection on CPU, and obtain a
new freeze and public seed. This attempt remains FAIL.
