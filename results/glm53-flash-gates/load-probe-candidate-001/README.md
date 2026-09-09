# Component-load probe candidate 1 / campaign round 40

Implementation9f37a677; fixed decision in configs/decision-specs/glm53-load-preflight.json.
Fixture RED41860ba0, probe REDa5effec1, controller RED39e76e77 are retained.
219 scoped CPU tests pass with the packaged isolated no-bytecode interpreter;
diff whitespace and secret-lint self-test pass. Diff manually scanned for canned
results, prompt detection, disabled assertions, timing manipulation and accidental
production activation. No production profile, service, engine or frozen helper
was changed. The probe is selected explicitly through --pinned-stream.

Review scope: scripts/42_probe_glm53_load.py, lib/glm53_load_fixture.py, their
focused tests and load-only additions to39_freeze/39_run_glm53_probe.py plus the
fixed load decision. Previously reviewed pinned-stream and replay helpers remain
byte-identical. The actual dense constructors were independently checked on meta
and all3,456 expert fixture names matched pinned static routing without CUDA.

The probe uses full component geometry, real selected loaders and finalizers,
byte checks before and after finalization, exact backing views and native pointer
tables, retained shared scratch and phase memory counters. Source payloads are
synthetic public-seed inputs retained locally with hashes; never model weights.
CPU scorer tests mock only generator digests to keep mutation tests bounded;
actual probe scoring independently regenerates every expected tensor digest.
No GPU measurement, full-model fit, model fidelity, context, speed or promotion
result is claimed. Existing internal pageable pointer-table copies remain visible
costs; the pinned claim applies to the selected tensor upload path only.
