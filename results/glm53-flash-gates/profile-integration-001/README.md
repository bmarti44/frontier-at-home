# Optional named profile integration

Acceptance fixed before implementation: both experimental profiles render the
exact four-slot geometry and containment; the production profile and launch
parameter overrides are rejected; changed/unlisted runtime files and stale
process identities fail closed. All tests must pass. These are lifecycle tests,
not model qualification or performance evidence.

`red.log` is the genuine unchanged-code result (four errors, missing profiles
and module) at 721041dd. Existing production admission/default tests must also
continue to pass. Start stays explicit, uses the existing hardened wrapper,
and never writes the production switch's active/default state.

Candidate 1 adds `cuda-spark-128g-agent-fast` and
`cuda-spark-128g-1m-experimental`, with start/status/stop through script93 and the
existing hardened wrapper. It does not change production admission or default
state. Profile arguments/environment are rendered once; containment must equal
the measured wrapper envelope. Startup verifies compressed, digest-bound closed
inventories, checks authentication/model/semantics, and emits `ready` only then.
Shutdown checks the exact systemd InvocationID and waits for unit/cgroup exit.

`green.txt` records ten passing configuration/lifecycle tests. The seventeen
existing GLM contract tests and fifteen resolver tests also pass. The actual
profile model launch is pending because server019 is running the direct context
replay. This is not production qualification or a model performance result.

The profile inventories extend freeze004 coverage to 48 existing runtime bytecode
files and 130 model metadata/receipt files that its original list omitted. See
`configs/build-manifests/glm53-local/coverage.json`. No executable source or native
binary was added or changed. The launcher rehashes the entire closed trees before
use; inherited hashes are not accepted as evidence of a successful launch. This
also limits the current replay to failure localization, not closed-inventory
confirmation. Its original freeze and inputs remain unchanged.

Candidate 1 review failed on three verified lifecycle defects (merged across the
two persistent reviewers): unknown/terminal systemd state skipped cleanup checks,
stop errors skipped controller waiting/handler restoration, and preparation
preceded registration/cancellation. `review-candidate-1.json` and
`review-red.txt` preserve the assertions and genuine failed regressions.

Candidate 2 targets those findings only. Registration/locking covers preparation;
identity-bound cancellation waits for the lifecycle lock; unit observation errors
fail closed; terminal states still check descendants; and nested cleanup retains
controller ownership and restores handlers even on stop errors. Fifteen focused
tests pass, including cancellation of a real small CPU preparation process before
any model launch. The existing seventeen GLM and fifteen resolver tests still pass.

Candidate 2 closed H1 and H3. Both reviewers retained H2: retries could still
escape on operational/logging errors, and wrapper exit could release ownership
without successful unit/cgroup observation. Candidate 3 changes only that retry
loop. It requires controller exit plus verified unit/cgroup cleanup, tolerates
operational stop errors and best-effort logging, and retains the lock while
observation is unavailable. Seventeen focused tests pass; raw failed/successful
results remain alongside both candidate-2 review records. The readiness cleanup
test now expects the required second check after controller exit.

Both persistent reviewers cleared candidate 3 (`review-candidate-3.json`). The
code/lifecycle gate passes. Both profile argv/env arrays were independently
compared with the recorded launches: agent-fast matches its serving snapshot;
1m-experimental matches server019 with CUDA_LAUNCH_BLOCKING removed. Actual
profile launch and full-model qualification remain pending.
