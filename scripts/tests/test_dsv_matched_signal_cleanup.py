#!/usr/bin/env python3
import os
import fcntl
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
            lock_fd = os.open(
                "/run/lock/frontier-at-home/inference.lock", os.O_RDWR
            )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                parent_pid = os.getpid()
                parent_ticks = Path(f"/proc/{parent_pid}/stat").read_text().split()[21]
                info = os.fstat(lock_fd)
                key = None
                for line in Path(f"/proc/{parent_pid}/fdinfo/{lock_fd}").read_text().splitlines():
                    fields = line.split()
                    if fields and fields[0] == "lock:":
                        key = fields[6]
                self.assertIsNotNone(key)
                environment = {
                    "HOME": "/home/bmarti44",
                    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "XDG_RUNTIME_DIR": "/run/user/1000",
                    "DSV4_MATCHED_KILL_FLOOR_GIB": "8",
                    "DSV4_MATCHED_MIN_START_GIB": "110",
                    "DSV4_MATCHED_MEMORY_HIGH_GIB": "105",
                    "DSV4_MATCHED_MEMORY_MAX_GIB": "107",
                    "DSV4_MATCHED_TIMEOUT_S": "5400",
                    "GLM_SAFE_PARENT_LOCK_PID": str(parent_pid),
                    "GLM_SAFE_PARENT_LOCK_START_TICKS": parent_ticks,
                    "GLM_SAFE_PARENT_LOCK_FD": str(lock_fd),
                    "GLM_SAFE_PARENT_LOCK_DEV_INO": f"{info.st_dev}:{info.st_ino}",
                    "GLM_SAFE_PARENT_LOCK_KERNEL_KEY": key,
                }
                process = subprocess.Popen(
                    [str(WRAPPER), "--tag", tag, "--", "/usr/bin/sleep", "30"],
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                unit = f"dsv4-matched-{tag}-{process.pid}.service"
                for _ in range(100):
                    active = subprocess.run(
                        ["systemctl", "--user", "is-active", unit],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    ).returncode == 0
                    if active:
                        break
                    time.sleep(0.05)
                self.assertTrue(active, f"signal={signum.name} unit never became active")
                process.send_signal(signum)
                returncode = process.wait(timeout=10)
                self.assertEqual(
                    returncode, 128 + signum.value, f"signal={signum.name} status"
                )
                self.assertNotEqual(
                    subprocess.run(
                        ["systemctl", "--user", "is-active", unit],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    ).returncode,
                    0,
                    f"signal={signum.name} wrapper did not stop its transient unit",
                )
            finally:
                self._stop_probe_units(tag)
                os.close(lock_fd)
                time.sleep(0.05)


if __name__ == "__main__":
    unittest.main()
