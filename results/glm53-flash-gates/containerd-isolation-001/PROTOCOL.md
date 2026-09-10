# Replay with Docker, its socket and containerd stopped

The owner completed the guarded containerd stop after Docker-only replay 012
failed before model admission. Independently verify all three services inactive,
the recorded containerd PID/start identity absent, GLM stopped, no listener on
8015 and stable available memory above 110 GiB. Record the new owner-installed
scoped runtime grant and actual noninteractive verification before the baseline.

Run one fresh replay of the unchanged full-context profile and closed durability
workload, following `docker-isolation-001/PROTOCOL.md`. The only environmental
difference from attempt 012 is that containerd is now stopped. The earlier global
swap failure and process correlation remain unchanged; stopping this service is
a hypothesis, not a proved fix. No other daemon stop or warmup variant is allowed
in this attempt. Finish Kimi generation and reference metadata work before the
fresh broad baseline. The new prompt corpus is not part of this closed workload.

Preserve four 262,144-token slots, the 1,048,576 aggregate cap, model/runtime,
startup flags, scheduler, fixed scorers, compiled-input binding, memory limits,
timeouts and authentication. Freeze clean source and configuration, obtain later
verified public randomness, and preserve broad global swap counters from before
freeze through preparation, startup, requests, stop and postchecks. Any observed
swap-in/out or used-swap change still fails admission or the admitted attempt.
Fresh containment, inference lock, external memory/identity guards and existing
900-second process/global census remain mandatory. Record its exact coverage.

Require authenticated native READY, short correctness, the fixed 20-request
necessary window, unchanged compiled inputs and every host prerequisite before
the 1,800-second workload. Stop immediately on a failed prerequisite; do not
start a workload after a failed preflight. Keep all failed attempts.

Use identity-verified profile shutdown and prove no survivors before restoring
the saved runtime service state. The newly installed exact `sudo -n` service
commands allow restoration without another owner password prompt; invoke each
service separately. Do not enable containerd or alter the existing default,
proxy, rollback settings or legacy guard breaker. Current-configuration direct
context, native paired fidelity, maximum media and production switching remain
separate gates even if this durability attempt passes.
