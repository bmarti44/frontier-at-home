# Docker-isolated replay of the unchanged full-context profile

The owner reports completing the reviewed Docker stop operation. Verify both
docker.service and docker.socket inactive, the old daemon identity gone, no
large model present and at least 110 GiB available. Record these observations;
do not infer completion from the owner's message alone. The preceding preflight
FAIL results and both unsuccessful environmental/startup branches remain intact.

This is one fresh replay under the [reviewed operator procedure](../../../docs/GLM53-DOCKER-ISOLATION.md).
Do not change model weights, runtime, startup flags, scheduler, context, fixture
generator, scorers, memory limits, timeouts, authentication or recorded default.
Keep four 262,144-token slots and the 1,048,576 aggregate cap. No additional
control warmup, daemon stop or privilege change is part of this candidate.

Use clean source, a new freeze and later verified public randomness. Preserve
the broad pre-freeze global baseline through requests, stop and post-run checks.
Any observed swap-in/out or used-swap change still rejects admission or fails an
admitted attempt. Fresh containment, inference lock, identity guard and external
memory watchdog remain mandatory. Admit the fixed 1,800-second workload only
after exact native startup/authentication/READY, smoke, the full 20-request
necessary window, unchanged compiled inputs and all host prerequisites pass.

Run the unchanged external process/global census for its existing 900-second
duration and record its actual coverage. Do not use the six-group observer that
requires Docker's now-absent cgroup. Retain global phase/terminal counters and
ongoing watchdog/identity samples beyond the census endpoint. Avoid optional
service queries while the model is loaded. Preserve any read errors and failure.

Stop GLM through its normal identity-verified profile interface on failure or
completion. Verify no surviving model process before restoring Docker's saved
active state. Docker restoration requires the owner's administrator access;
do not expand delegated controls. This run cannot promote GLM or establish the
separate current-configuration direct-context, fidelity, maximum-media or
production-switching gates merely by passing durability.
