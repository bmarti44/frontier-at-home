# Native attempt 004 — FAIL

The frozen run used public beacon round 6450768. All fourteen synthetic native
assertions and the child identity guard completed, but the outer controller
failed because this packaged Python lacks `os.pidfd_open`. Consequently no live
unit observation was recorded and the combined attempt is FAIL. The wrapper
returned zero, the exact unit is inactive and its cgroup is absent. Runtime and
frozen-file postchecks completed; no model weights were loaded. Do not treat the
inner successes as a complete native, fidelity, context or performance result.

The exact source crash directory remains in capture.json. Code, manifest, seed,
raw logs and nested assertions/identity evidence are copied byte-for-byte. JIT
state stays in the local attempt directory and is not qualification evidence.
Next fix is the observed packaged-interpreter pidfd API compatibility gap,
including its guard error-cleanup path; run CPU process controls using the actual
packaged interpreter before another frozen hardware attempt.
