# Proposed temporary host swap pause — not executed

The existing global swap gate has failed with Docker alone stopped, with Docker
and containerd stopped, and during the native layer probe. The latest failure
wrote five pages (20 KiB) while more than 100 GiB remained available. A wider
retrospective interval accounts five writes to system.slice, with no increase
in user.slice, but identifies neither a specific service nor the causal trigger.
Further daemon-stop and warmup variants are not proposed.

The concrete next prerequisite is to temporarily deactivate only the existing
`/swap.img` swap unit while no large model is running and at least 110 GiB is
available. The latest observation showed 146,764 KiB swap used, approximately
143 MiB, and about 115 GiB available. Recheck those conditions immediately before
the root operation. This proposal creates no new sudoers grant and changes no
profile, default, fstab entry or persistent service enablement.

Owner command:

```bash
sudo /usr/bin/systemctl stop swap.img.swap
```

Require the command to succeed, the exact unit to be inactive, `/proc/swaps` to
contain no active swap entries and `/proc/meminfo` to show zero swap total/free.
An error is a failed preparation. Do not silently continue. Only then establish
a new broad host baseline, stable start memory, freeze and later public seed.
Keep the existing global swap, memory floor, OOM/Xid, timeout, identity and
cleanup checks unchanged. Prior failed attempts remain FAIL. This is not itself
a model qualification or a guarantee that native reference generation will fit
the time budget; the first shard consumed about 215 seconds of a 600-second probe.

After terminal and postartifact qualification observations, restore the same unit:

```bash
sudo /usr/bin/systemctl start swap.img.swap
```

This action is outside the eight installed passwordless Docker/containerd
commands. AGENTS.md requires explicit owner authorization for new privilege.
No swap state has been changed by the agent and Codex is not run as root.
