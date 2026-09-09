# Identity gate paused — local correction prepared

The last submitted identity candidate is `6beaa5ff`, candidate 2 at campaign
review round 19. Both persistent reviewers reproduced the same high I1 finding:
an `atexit` callback can replace Python with `/bin/true` after the final
completion handshake and still receive PASS. Candidate 1 and candidate 2 each
retain one high finding. AGENTS.md lines 104–108 require pausing this gate.
The separate medium cleanup-anchor finding is closed.

The uncommitted patch in `proposed-local.patch` installs a terminal-only seccomp
filter after the frozen probe returns and before the final blocked identity
sample. It prohibits executable replacement through `execve` and `execveat`,
synchronizes the filter across existing threads, and verifies the added filter
and no-new-privileges state from `/proc`. Interpreter finalization remains normal;
an attempted replacement terminates the probe and fails its verdict. The
existing GLM cgroup wrapper, serving profiles and startup default are unchanged.
The kernel semantics are documented by the [seccomp manual](https://man7.org/linux/man-pages/man2/seccomp.2.html)
and [kernel filter documentation](https://www.kernel.org/doc/html/v5.15/userspace-api/seccomp_filter.html).
AArch64 syscall numbers and flags were checked against this host's Linux UAPI headers.

Local CPU regression results: all 141 combined checks pass, including 12 real
process/identity controls. The exact `atexit` bypass now fails; terminal
`execveat` and an existing thread's replacement attempt terminate with SIGSYS;
ordinary exit callbacks still execute. This is a prepared correction, not a
reviewed or frozen native candidate. No reviewer round or GPU run was started
with this patch. The two edited source/test files intentionally remain uncommitted.

Before hardware qualification resumes, the identity gate needs owner direction,
then persistent-reviewer closure and a new source/runtime/configuration freeze
with fresh public randomness. The outer attempt verdict must also bind the
identity evidence to raw cgroup, memory, swap, kernel and cleanup observations.

Current integration facts:

- Optional estimated profile: 1,048,576 aggregate tokens in four 262,144-token slots.
- All three pinned native components built; packaged runtime inventories retained.
- Native attempt 003 passed 14 synthetic correctness assertions. Continuous
  Python identity for attempts 001–003 remains explicitly NO_RESULT.
- BF16 mixed-shard constructor fix is default off and passed both reviews.
- The cache allocation scorer closed its physical-ID finding in candidate 2,
  campaign round 17. Its 9,565,304,320-byte allocation remains unmeasured.
- No GLM weight payload has been downloaded or loaded. No serving/default change.
- Full-model fidelity, four-slot occupancy, multimodal behavior, runtime JIT
  closure, service lifecycle, authentication/switching/rollback and performance
  gates remain pending.
