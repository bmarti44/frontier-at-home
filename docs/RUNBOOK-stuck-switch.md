# Runbook: engine switch hangs or deadlocks

For when `scripts/52_engine_switch.sh` (any verb) does not return, or a
gate window cannot open. Written after the 2026-08-21 incident: a leaked
lock fd caused every switch invocation to hang silently in `flock` for
hours while production stayed healthy.

## 1. Confirm what is actually wrong

```
sudo scripts/52_engine_switch.sh status        # active profile record
curl -s http://127.0.0.1:8013/health           # production health (direct)
pgrep -af 52_engine_switch                     # queued/blocked invocations
ps -o pid,ppid,etime,cmd -C flock              # blocked lock waiters + ages
```

A healthy production with old (`etime` in minutes/hours) `flock -x 9`
children means the switch lock is stuck, not the engine.

## 2. Identify the lock holder

```
stat -c %i /home/dsv4/ds4-project/engine-switch/switch.lock   # inode
grep <inode> /proc/locks
```

Lines without `->` are holders; `->` lines are waiters. The 5th field is
the PID **that created the lock**, which may be dead: flock belongs to
the open-file description, so it survives in any child that inherited
the fd. If the holder PID is not running, the lock lives on in a spawned
long-lived child — historically the profile memwatch
(`pgrep -af 01_memwatch`). `lslocks` shows the same picture and prints
`(unknown)` for dead creator PIDs.

The switch now acquires with a bounded wait
(`SWITCH_LOCK_TIMEOUT_SECONDS`, default 300) and prints this diagnosis
itself on timeout; this section is for confirming it.

## 3. Drain safely — order matters

1. **Kill queued switch invocations first.** A blocked invocation holds
   an open fd to the script *text it started with*; if the script was
   fixed on disk meanwhile, the queued copy still runs the old code when
   it unblocks. `sudo pkill -f 52_engine_switch.sh` (exactly this form —
   it is the granted passwordless command) kills every queued invocation.
2. **Release the leaked lock by ending its holder.** If the holder is
   the production memwatch, do NOT kill it directly — it exits on its
   own when its armed target dies. Stop the engine instead:
   `sudo systemctl stop qwen38-engine.service` (or the profile's unit).
   The memwatch observes target death, exits fail-closed, and the lock
   releases.
3. **Restore production with the on-disk (fixed) script:**
   `sudo scripts/52_engine_switch.sh restore`. Verify:
   `curl -s http://127.0.0.1:8013/health` and
   `grep <inode> /proc/locks` → no holders.

## 4. Rules that prevent recurrence

- Every backgrounded spawn in the switch closes fd 9 (`9>&-`);
  `test_engine_switch.py::test_background_spawns_close_the_switch_lock_fd`
  enforces it. The serve scripts already follow the same rule.
- Never edit a script that a queued/running process may execute —
  install changes by copy + atomic `mv` (the running process keeps its
  old inode; the rename cannot corrupt it).
- Direct-execution scripts keep their exec bits
  (`test_executable_modes.py`); a lost bit on `03_memory_guard.py` is a
  known restore-breaker.
- Gate windows open with `sudo scripts/52_engine_switch.sh stop`
  (leaves `active.json`, so `restore` brings the same profile back).
