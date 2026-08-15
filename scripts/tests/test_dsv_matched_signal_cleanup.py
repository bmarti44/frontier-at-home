#!/usr/bin/env python3
import os
import signal
import subprocess
import time
import unittest
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "results/glm52-gates/harness/dsv4_matched_cgroup_run.sh"


class DsvMatchedSignalCleanupTests(unittest.TestCase):
    def _stop_probe_units(self, tag: str):
        listing = subprocess.run(
            [
                "systemctl", "--user", "list-units", "--all", "--plain",
                "--no-legend", f"dsv4-matched-{tag}-*",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for line in listing.stdout.splitlines():
            unit = line.split(maxsplit=1)[0]
            if unit.endswith(".service"):
                subprocess.run(
                    ["systemctl", "--user", "stop", unit],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )

    def test_term_int_and_hup_stop_the_exact_transient_unit(self):
        if subprocess.run(
            ["systemctl", "--user", "show-environment"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode != 0:
            self.skipTest("user systemd is unavailable")
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            tag = f"c6-sig-{signum.value}-{uuid.uuid4().hex[:8]}"
            script = r'''
set -Eeuo pipefail
wrapper=$1
tag=$2
signal_number=$3
exec {lock_fd}<>/run/lock/frontier-at-home/inference.lock
flock -n "$lock_fd"
parent_pid=$$
parent_ticks=$(awk '{print $22}' "/proc/$$/stat")
parent_devino=$(stat -Lc '%d:%i' "/proc/$$/fd/$lock_fd")
parent_key=$(awk '$1 == "lock:" && $3 == "FLOCK" && $5 == "WRITE" {print $7}' "/proc/$$/fdinfo/$lock_fd")
env DSV4_MATCHED_KILL_FLOOR_GIB=8 DSV4_MATCHED_MIN_START_GIB=110 \
    DSV4_MATCHED_MEMORY_HIGH_GIB=105 DSV4_MATCHED_MEMORY_MAX_GIB=107 \
    DSV4_MATCHED_TIMEOUT_S=5400 GLM_SAFE_PARENT_LOCK_PID="$parent_pid" \
    GLM_SAFE_PARENT_LOCK_START_TICKS="$parent_ticks" \
    GLM_SAFE_PARENT_LOCK_FD="$lock_fd" GLM_SAFE_PARENT_LOCK_DEV_INO="$parent_devino" \
    GLM_SAFE_PARENT_LOCK_KERNEL_KEY="$parent_key" \
    "$wrapper" --tag "$tag" -- /usr/bin/sleep 30 &
wrapper_pid=$!
unit="dsv4-matched-${tag}-${wrapper_pid}.service"
for _ in $(seq 1 100); do
    [[ $(systemctl --user is-active "$unit" 2>/dev/null || true) == active ]] && break
    sleep 0.05
done
[[ $(systemctl --user is-active "$unit" 2>/dev/null || true) == active ]]
kill -"$signal_number" "$wrapper_pid"
set +e
wait "$wrapper_pid"
wrapper_rc=$?
set -e
[[ $wrapper_rc == $((128 + signal_number)) ]]
for _ in $(seq 1 100); do
    [[ $(systemctl --user is-active "$unit" 2>/dev/null || true) != active ]] && break
    sleep 0.05
done
[[ $(systemctl --user is-active "$unit" 2>/dev/null || true) != active ]]
'''
            try:
                result = subprocess.run(
                    ["bash", "-c", script, "signal-test", str(WRAPPER), tag, str(signum.value)],
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=20,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"signal={signum.name} wrapper did not stop its transient unit",
                )
            finally:
                self._stop_probe_units(tag)
                time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
