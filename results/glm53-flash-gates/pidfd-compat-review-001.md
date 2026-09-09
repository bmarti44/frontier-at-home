# Packaged-interpreter pidfd compatibility closure

Candidate 1 / campaign round 28, source `c65e6caa`, closes the execution defect
found in native-004. Both persistent reviewers report zero high or critical
findings. All 26 affected guard/capture/pidfd tests pass independently using the
actual packaged interpreter; the full author CPU audit passed 153 tests there.

The typed libc entry points match installed headers, produce close-on-exec pidfds,
preserve EBADF and ESRCH/ProcessLookupError, and reject integer overflow. Independent
checks confirmed descriptor-based signaling reaches the intended child. The mapped
libc matches the newly frozen library path. No numeric-PID signaling fallback is
used. The source guard/capture APIs otherwise retain their existing contracts.

Native-004 remains a complete failed attempt. A new source/configuration freeze
and new predetermined public beacon are required for native-005. No GPU was used
during compatibility review and no model has been loaded or qualified.
