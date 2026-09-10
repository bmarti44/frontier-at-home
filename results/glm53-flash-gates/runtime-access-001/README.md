# Owner-installed scoped runtime access

The optional installer was prepared with genuine RED evidence, three passing
checks, shell syntax validation and two scoped reviews. `audit.tar.gz` retains
that pre-installation state and explicitly records that installation had not yet
occurred at review time.

The owner subsequently ran the installer. `installed-verification.json` records
the live sudo policy and an actual successful `sudo -n` invocation that stops the
already inactive containerd service under the inference lock. Docker, its socket,
containerd and GLM were stopped before that idempotent check. No model started.
The exact runtime operations can now run without another password prompt; the
existing identity, task, memory and restoration checks still apply.
