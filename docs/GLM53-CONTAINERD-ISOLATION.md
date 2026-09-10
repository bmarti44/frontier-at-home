# Proposed containerd isolation after the Docker-isolated preflight

Status: the owner completed this stop and installed the separate
[scoped runtime grant](GLM53-RUNTIME-ACCESS.md). All three runtime services are
inactive; actual noninteractive operation was verified. The procedure below is
the historical, identity-bound owner operation and should not be rerun with its
old PID. Its pre-action bytes and reviews remain in attempt 012's evidence.

Docker and docker.socket are stopped as requested by the owner. GLM was never
loaded in attempt 012. Its preflight failed on one swap-in page after fixture
preparation. The process census recorded containerd PID 2120/start ticks 1002
losing 4 KiB of VmSwap and gaining one major fault in the same interval. This is
a correlation, not complete attribution or a waiver of the failed host gate.

The native GLM backend does not use Docker or containerd. Read-only inspection
found no containerd-shim/Kubernetes processes, no named containerd socket clients,
and only the inactive Docker service in containerd's reverse dependencies.
Containerd remains active but disabled for boot; its unit has Restart=always and
KillMode=process. Stopping Docker did not stop this separate service.

The final missing consumer check is the running-task list in every containerd
namespace. Access to its socket is restricted to root. The owner can perform
that check and the proposed stop in one guarded administrative command below.
It aborts on a task-query failure, running tasks, changed daemon identity, an
active Docker unit or a held inference lock. No delegated privilege or service
configuration is added. Finish the current observation and repeat the read-only
consumer checks before using it; unknown or newly active consumers block this
intervention. Do not kill the daemon PID directly.

```bash
sudo bash -eu <<'SH'
exec 9</run/lock/frontier-at-home/inference.lock
flock -n 9
verify_services() {
    test "$(systemctl show docker.service -p ActiveState --value)" = inactive
    test "$(systemctl show docker.socket -p ActiveState --value)" = inactive
    test "$(systemctl show containerd.service -p MainPID --value)" = 2120
    test "$(awk '{print $22}' /proc/2120/stat)" = 1002
    test "$(systemctl show containerd.service -p InvocationID --value)" = 67c0d0a5eed647148a596752d5d046cb
}
verify_services
namespaces=$(timeout 30s ctr namespaces list -q)
while IFS= read -r ns; do
    [ -n "$ns" ] || continue
    tasks=$(timeout 30s ctr --namespace "$ns" tasks list -q)
    [ -z "$tasks" ] || { echo "Running tasks in $ns; containerd left active."; exit 1; }
done <<< "$namespaces"
verify_services
systemctl stop containerd.service
SH
```

Afterwards verify the service inactive and its old process identity absent before
creating a new frozen attempt and broad baseline. Keep the full context, model,
scorers and all host gates unchanged. A failed prerequisite still prevents model
admission. The earlier Docker-only failure remains preserved.

After testing, stop GLM through its identity-verified profile command and confirm
no survivors before the owner restores the previously active runtime services:

```bash
sudo systemctl start containerd.service docker.socket docker.service
```

Do not enable containerd or reset the legacy guard breaker. The existing
[privilege rule](../AGENTS.md) prohibits expanding delegated controls; this uses
the owner's existing administrator access. No model qualification, cold-boot,
fidelity, media or performance claim follows merely from stopping a service.
