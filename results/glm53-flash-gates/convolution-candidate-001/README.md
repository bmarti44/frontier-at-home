Convolution candidate 1 / campaign round 43
==========================================

Implementation c074779f. Test-before-implementation 9a1d46c5; genuine RED and additional fresh length-1 coverage d8660409. All 229 scoped CPU tests pass with packaged Python -I -B. Diff whitespace and secret scans pass. Diff inspection found no prompt detection, canned production results, disabled assertions, timing manipulation or production activation.

Scope: new 44_probe_glm53_conv.py, its CPU tests and fixed decision spec; conv-only controller/freezer routing. Existing signed-off probe APIs, safety, capture, host scoring and replay helpers are unchanged.

Eight public-seed-permuted cases use actual pinned causal_conv1d_fn/update functions, selected SD state view, qkv strided slice and FP32 taps. Independent CPU array oracle covers every output element and byte-exact final state including inactive/null slots. Captured post-input bytes verify all prefill inputs and decode output/tail invariants; actual storage alias is checked separately. Persistent pinned probe buffers and retained upstream pinned metadata feed GPU transfers. Mutation checks reject malformed case coverage, aliases, strides, artifacts, timestamps, counters, state and input bytes. CPU synthetic fixtures are not qualification evidence.

No model weights or GPU kernels have been run for this candidate. The first contained run is JIT preparation and can only earn NO_RESULT overall even when kernel/host checks pass. Sealed replay will be a later frozen candidate. No full-model memory, fidelity, context or performance claim; optional model admission stays closed.
