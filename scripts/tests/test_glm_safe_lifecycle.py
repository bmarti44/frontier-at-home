#!/usr/bin/env python3
"""No-model integration mutations for candidate provenance shutdown."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
FIXTURE = ROOT / "scripts/tests/fixtures/candidate_lifecycle.c"


class CandidateLifecycleSourceTests(unittest.TestCase):
    def test_disappearing_proc_counters_are_emitted_as_numeric_zero(self):
        source = SAFE.read_text(encoding="utf-8")
        start = source.index("  RSS=$(awk ")
        end = source.index('  echo "$(date -u --iso-8601=ns)', start)
        sampler = source[start:end]
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / "status").write_text("", encoding="utf-8")
            (proc / "io").write_text("", encoding="utf-8")
            sampler = sampler.replace(
                '"/proc/$SPID2/status"', f'"{proc / "status"}"'
            ).replace(
                '"/proc/$SPID2/io"', f'"{proc / "io"}"'
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -u\nSPID2=missing\nREQUIRE_CGROUP=0\n"
                    + sampler
                    + 'printf "%s|%s\\n" "$RSS" "$RB"\n',
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0|0\n")

    def test_exit_race_enters_shutdown_grace_before_identity_failure(self):
        source = SAFE.read_text(encoding="utf-8")
        shutdown = source.index("if [[ -z $CURRENT_STATE")
        identity = source.index(
            'plog "FATAL executed candidate identity changed pid=$EXECUTED_PID',
            shutdown,
        )
        guarded = source[shutdown:identity]
        self.assertIn("-z $CURRENT_HASH", guarded)
        self.assertIn("-z $CURRENT_DEVICE_INODE", guarded)

    def test_verified_candidate_environment_is_hash_bound_from_proc(self):
        source = SAFE.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_PROVENANCE_ENV_ALLOWLIST", source)
        self.assertIn("GLM_SAFE_EXPECTED_ENV_SHA256", source)
        self.assertIn('"/proc/$SPID2/environ"', source)
        self.assertIn("executed_environment_sha256=", source)
        self.assertIn("executed candidate environment mismatch", source)

    def test_default_off_current_user_mode_retains_containment_and_provenance(self):
        safe = SAFE.read_text(encoding="utf-8")
        launcher = CGROUP.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER", safe)
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER", launcher)
        self.assertIn("/home/bmarti44/.local/state/glm52-crashlog", safe)
        self.assertIn("/home/bmarti44/.cache/glm52-", safe)
        self.assertIn("MemorySwapMax=0", launcher)
        self.assertIn("OOMPolicy=kill", launcher)
        self.assertIn("GLM_SAFE_PROVENANCE_ENV_ALLOWLIST", launcher)
        self.assertIn("GLM_SAFE_EXPECTED_ENV_SHA256", launcher)
        self.assertIn("/run/user/$UID/glm52-inference.lock", launcher)
        self.assertIn("/usr/bin/sudo -n -u dsv4", launcher)
        self.assertIn('RUN_CWD=$(pwd -P)', launcher)
        self.assertIn('--working-directory="$RUN_CWD"', launcher)
        self.assertIn("GLM_SAFE_WITNESS_NONCE", safe)
        self.assertIn("glm52-w1-witness", safe)
        self.assertIn("GLM_SAFE_WITNESS_NONCE", launcher)
        self.assertIn("GLM_SAFE_WITNESS_ARTIFACT", safe)
        self.assertIn("artifact_sha256=", safe)
        self.assertIn("GLM_SAFE_WITNESS_ARTIFACT", launcher)
        self.assertIn("date -u --iso-8601=ns", safe)

    def test_authoritative_timestamps_are_emitted_in_utc(self):
        source = SAFE.read_text(encoding="utf-8")
        self.assertIn(
            'plog() { echo "$(date -u --iso-8601=ns) $*" >> "$MAIN"',
            source,
        )
        self.assertIn(
            'echo "$(date -u --iso-8601=ns) mem_avail_kb=$MA',
            source,
        )
        self.assertNotIn("date -u -Is", source)
        self.assertNotIn('echo "$(date -Is)', source)
        self.assertNotIn('echo "$(date --iso-8601=ns)', source)

    def test_cgroup_and_xid_safety_telemetry_is_preserved(self):
        source = SAFE.read_text(encoding="utf-8")
        for control in (
            "memory.current",
            "memory.peak",
            "memory.swap.current",
            "memory.events.local",
        ):
            self.assertIn(control, source)
        self.assertIn("cgroup_current_bytes=", source)
        self.assertIn("cgroup_peak_bytes=", source)
        self.assertIn("cgroup_swap_current_bytes=", source)
        self.assertIn('KERNEL_LOG="$DIR/kernel.log"', source)
        self.assertIn("NVRM.*Xid", source)
        self.assertIn("FATAL kernel Xid evidence appeared during run", source)


class CandidateLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc is unavailable")
        probe = subprocess.run(
            ["sudo", "-n", "-u", "dsv4", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode:
            raise unittest.SkipTest("passwordless dsv4 test account is unavailable")
        available_gib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        ) / 1048576
        if available_gib < 110:
            raise unittest.SkipTest("110 GiB safe-run precondition is unavailable")

        cls.local_tmp = Path(tempfile.mkdtemp(prefix="glm-lifecycle-"))
        os.chmod(cls.local_tmp, 0o755)
        cls.runner = cls.local_tmp / "runner"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-o", cls.runner, FIXTURE],
            check=True,
        )
        os.chmod(cls.runner, 0o755)
        created = subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "mktemp", "-d",
                "/home/dsv4/ds4-project/src/glm-safe-lifecycle.XXXXXX",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        cls.candidate_src = Path(created.stdout.strip())
        subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "cp", "--",
                "/usr/bin/sleep", cls.candidate_src / "ds4-server",
            ],
            check=True,
        )
        cls.digest = hashlib.sha256(
            Path("/usr/bin/sleep").read_bytes()
        ).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "candidate_src"):
            target = str(cls.candidate_src)
            if target.startswith(
                "/home/dsv4/ds4-project/src/glm-safe-lifecycle."
            ):
                subprocess.run(
                    ["sudo", "-n", "-u", "dsv4", "rm", "-rf", "--", target],
                    check=False,
                )
        if hasattr(cls, "local_tmp"):
            shutil.rmtree(cls.local_tmp)

    def run_mutation(self, mode: str) -> subprocess.CompletedProcess[str]:
        environment = {
            "HOME": "/home/dsv4",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "GLM_CANDIDATE_SRC": str(self.candidate_src),
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": self.digest,
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "30",
        }
        return subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "env",
                *[f"{key}={value}" for key, value in environment.items()],
                "bash", str(SAFE), "--tag", f"lifecycle-{mode}", "--",
                str(self.runner), mode, str(self.candidate_src / "ds4-server"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )

    def test_clean_candidate_zombie_during_wrapper_exit_is_accepted(self):
        result = self.run_mutation("clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_replacement_candidate_during_exit_is_rejected(self):
        result = self.run_mutation("replace")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonzero_wrapper_after_candidate_exit_is_rejected(self):
        result = self.run_mutation("fail")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrapper_exceeding_shutdown_grace_is_rejected(self):
        result = self.run_mutation("linger")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)


class CurrentUserTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc is unavailable")
        available_gib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        ) / 1048576
        if available_gib < 110:
            raise unittest.SkipTest("110 GiB safe-run precondition is unavailable")

        cls.local_tmp = Path(tempfile.mkdtemp(prefix="glm-utc-lifecycle-"))
        cls.runner = cls.local_tmp / "runner"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-o", cls.runner, FIXTURE],
            check=True,
        )
        cache = Path("/home/bmarti44/.cache")
        cache.mkdir(parents=True, exist_ok=True)
        cls.candidate_src = Path(
            tempfile.mkdtemp(prefix="glm52-utc-lifecycle-", dir=cache)
        )
        shutil.copy2("/usr/bin/sleep", cls.candidate_src / "ds4-server")
        cls.digest = hashlib.sha256(
            (cls.candidate_src / "ds4-server").read_bytes()
        ).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "candidate_src"):
            shutil.rmtree(cls.candidate_src)
        if hasattr(cls, "local_tmp"):
            shutil.rmtree(cls.local_tmp)

    def test_real_wrapper_emits_only_utc_under_host_timezone_variants(self):
        for timezone_name in (
            "America/New_York",
            "UTC",
            "Pacific/Kiritimati",
        ):
            with self.subTest(timezone_name=timezone_name):
                environment = {
                    "HOME": "/home/bmarti44",
                    "PATH": (
                        "/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                        "/usr/bin:/sbin:/bin"
                    ),
                    "TZ": timezone_name,
                    "GLM_CANDIDATE_SRC": str(self.candidate_src),
                    "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
                    "GLM_SAFE_EXPECTED_BINARY_SHA256": self.digest,
                    "GLM_SAFE_KILL_FLOOR_GIB": "40",
                    "GLM_SAFE_MIN_START_GIB": "110",
                    "GLM_SAFE_TIMEOUT_S": "30",
                    "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
                }
                result = subprocess.run(
                    [
                        "env",
                        *[
                            f"{key}={value}"
                            for key, value in environment.items()
                        ],
                        "bash",
                        str(SAFE),
                        "--tag",
                        f"utc-{timezone_name.replace('/', '-')}",
                        "--",
                        str(self.runner),
                        "clean",
                        str(self.candidate_src / "ds4-server"),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=45,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                matches = re.findall(
                    r"SAFE_RUN_DONE rc=0 killed=no dir=(\S+)",
                    result.stdout,
                )
                self.assertEqual(
                    len(matches), 1, result.stdout + result.stderr
                )
                main = (Path(matches[0]) / "main.log").read_text()
                samples = (Path(matches[0]) / "samples.log").read_text()
                lifecycle = [
                    line.split()[0]
                    for line in main.splitlines()
                    if "executed_candidate_verified" in line
                    or "SAFE_RUN end" in line
                ]
                self.assertEqual(len(lifecycle), 2, main)
                self.assertTrue(
                    all(value.endswith("+00:00") for value in lifecycle),
                    lifecycle,
                )
                self.assertTrue(
                    all(
                        re.search(r"[.,]\d+\+00:00$", value)
                        for value in lifecycle
                    ),
                    lifecycle,
                )
                sample_timestamps = [
                    line.split()[0] for line in samples.splitlines() if line
                ]
                self.assertGreaterEqual(len(sample_timestamps), 2, samples)
                self.assertTrue(
                    all(
                        re.fullmatch(
                            r"\S+ mem_avail_kb=\d+ eng_rss_kb=\d+ "
                            r"read_bytes=\d+ "
                            r"cgroup_current_bytes=(?:\d+|na) "
                            r"cgroup_peak_bytes=(?:\d+|na) "
                            r"cgroup_swap_current_bytes=(?:\d+|na)",
                            line,
                        )
                        for line in samples.splitlines()
                    ),
                    samples,
                )
                self.assertTrue(
                    all(
                        value.endswith("+00:00")
                        for value in sample_timestamps
                    ),
                    sample_timestamps,
                )


if __name__ == "__main__":
    unittest.main()
