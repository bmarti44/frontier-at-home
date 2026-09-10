# Optional passwordless runtime controls

The owner requested removal of repeated password prompts for GLM qualification.
This is a separate, optional grant for `bmarti44`, installed once by the owner.
Its fixed acceptance contract is: only start/stop for `docker.service`,
`docker.socket`, and `containerd.service`, plus read-only containerd namespace
and task-ID listing. Task listing accepts one namespace made of letters, digits,
underscores, dots and hyphens, starting with a letter, digit or underscore.
All other executable/argument combinations are outside this new grant.

The installation must validate the drop-in with the host's `visudo`, write it as
root:root mode 0440, preserve unrelated sudoers rules, refuse to replace a different
existing grant, and perform no service actions. Installing the grant is separate
from starting or stopping any service. The inference lock, identity checks,
all-namespace task check, memory gates and shutdown-before-restore ordering still
apply. A task query that cannot complete must block isolation.

This grant supplements existing host permissions; its negative cases describe
what this new rule allows, not a revocation of the owner's existing administrator
access. It grants no command shell or writable-repository program execution as
root. Codex and the inference runner remain ordinary user processes.

The installed sudo 1.9.15 supports anchored POSIX argument expressions (introduced
in sudo 1.9.10); the installed `sudoers(5)` manual documents their matching and
the `NOPASSWD` tag. The acceptance tests validate the real sudoers parser and
exercise argument-boundary negatives before any privileged installation.

Install once from this checkout:

```bash
sudo bash scripts/dev/grant_glm53_runtime_once.sh
```

After installation the runner can use commands such as
`sudo -n /usr/bin/ctr namespaces list -q` and
`sudo -n /usr/bin/systemctl stop containerd.service`. Service operations remain
subject to the host checks above; the grant alone does not perform them. Call
each service command separately because the grant matches exact arguments.
Read-only `systemctl show` and profile status do not need sudo.

To remove only this new grant later:

```bash
sudo rm /etc/sudoers.d/glm53-runtime-control
```
