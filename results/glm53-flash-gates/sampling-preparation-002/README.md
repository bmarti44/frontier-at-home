# Sampling preparation: NO_RESULT

Intentionally stopped after discovering FLASHINFER_JIT_VERBOSE=1 also selects
CUDA debug compilation unless FLASHINFER_JIT_DEBUG is explicit. No model was
loaded. The next fresh build sets FLASHINFER_JIT_DEBUG=0 and retains two jobs,
the same source, and the same build/load acceptance.
