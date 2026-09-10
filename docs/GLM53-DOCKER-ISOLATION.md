# Proposed Docker isolation for GLM qualification

Status: the owner stopped Docker and its socket. The subsequent
[unloaded replay 012](../results/glm53-flash-gates/soak-native-012/README.md) still
failed its host swap gate. The owner later completed the separate
[guarded containerd stop](GLM53-CONTAINERD-ISOLATION.md).
The procedure below records the earlier Docker-only intervention. Its original
pre-action bytes are retained in the attempt-012 freeze.

The second unloaded preflight recorded one global swap-in page between
1789044832.2367342 and 1789044833.2368898. The same sample interval recorded one
page in Docker's cgroup accounting. GLM was never started. This identifies the
accounting owner, not the trigger for the read. The earlier failures remain FAIL.
Subsequent inspection found zero running containers and both docker.service and
docker.socket active. Docker's recorded PID was 2483, start ticks 1078.

The next bounded intervention is to stop Docker and its activation socket for
one fresh qualification attempt. The current GLM backend runs directly in its
own user systemd cgroup and does not require Docker. Keep model settings, the
broad no-swap gate, memory limits, authentication and recorded default unchanged.
This is a new service-isolation hypothesis, not another startup or control-warmup
variant. The two control/preflight alternatives remain NO_RESULT if both fail.

Before stopping anything, finish the current passive observer; recheck that GLM
is stopped, Docker still has no running containers, and the Docker unit/PID/start
identity matches the fresh observation. If containers have appeared, stop here.
Record both units' active/enabled state. Do not disable units, edit service files,
change sudoers or reset the legacy guard's existing circuit breaker.

The owner-only stop operation is:

```bash
sudo systemctl stop docker.socket docker.service
```

Verify both units inactive and the old daemon identity gone before establishing
the fresh pre-freeze baseline. Run the normal frozen GLM profile and existing
admission/scoring pipeline under containment. Use the existing process/global
observer; the six-group accounting observer requires a live Docker cgroup and
must not report an absent, intentionally stopped group as a zero counter.
Any subsequent host paging still fails the unchanged gate.

Stop GLM through its identity-verified profile interface and verify no surviving
model process before restoring Docker. This order also avoids overlapping model
work with Docker's NVIDIA-related startup dependencies. Restore the previously
active units, then verify their state and unchanged default/proxy/guard:

```bash
sudo systemctl start docker.socket docker.service
```

The installed passwordless controls cover model endpoints, not Docker service
control. A noninteractive privileged Docker status check returned “a password is
required.” The repository's [privilege rule](../AGENTS.md) says: “Never broaden
those controls or run Codex itself as root.” These two administrative operations
therefore need the owner's existing administrator access. No permanent privilege
change is proposed. This intervention would not itself qualify fidelity, media,
direct context, sustained speed, production switching or cold-boot behavior.
