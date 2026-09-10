# Orderly profile shutdown

The actual profile-launch-001 failure is committed in87c7cd1a. Both persistent
reviewers require the clean-lifecycle result to remain FAIL until the guard can
complete its existing terminal handshake. Do not suppress its BrokenPipe error
or weaken the frozen guard protocol.

The fixed CPU regression runs the real guard around a real SIGTERM-clean child,
then invokes the profile stop function. The child must complete before any
whole-unit fallback, guard exit must be0 with PASS, and raw identity must contain
one verified completion and an empty cleanup group. The two profiles must select
the existing bounded native shutdown timeout of10 seconds. No GPU is loaded.
