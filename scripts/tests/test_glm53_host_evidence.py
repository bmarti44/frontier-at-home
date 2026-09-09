"""Synthetic host-evidence mutations; fixtures are not qualification results."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from glm53_host_evidence import score_host_observations, unit_query


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


class HostEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "identity").mkdir()
        self.base = 1700000000.0
        self.unit = "glm52-glm53-native-smoke-004-321.service"
        self.cg = "/user.slice/user-1000.slice/user@1000.service/app.slice/" + self.unit
        self.expected = {"binary_sha256": "a" * 64, "executable": "/runtime/bin/python3", "guard": "/code/guard.py",
                         "guard_sha256": "b" * 64, "probe": "/code/probe.py", "probe_sha256": "c" * 64,
                         "probe_arguments": ["--seed", "1"], "environment_sha256": "d" * 64,
                         "unit_prefix": "glm52-glm53-native-smoke-004-", "kill_floor_gib": 40,
                         "minimum_start_gib": 110, "timeout_seconds": 600, "maximum_sample_gap_seconds": 2.0}
        manifest = {"qualification": "Python_probe_identity_only", "executable": self.expected["executable"],
                    "binary_sha256": self.expected["binary_sha256"], "device_inode": [100, 200], "cgroup": "0::" + self.cg + "\n",
                    "argv": ["/runtime/bin/python3", "-I", "-B", "/code/guard.py", "--child", "4", "5", "/code/probe.py", "--seed", "1"],
                    "environment_sha256": self.expected["environment_sha256"], "period_seconds": 0.25, "start_unix": self.base - 0.2,
                    "source_files": [{"path": "/code/guard.py", "sha256": "b" * 64}, {"path": "/code/probe.py", "sha256": "c" * 64}]}
        write_json(self.root / "identity/manifest.json", manifest)
        identities = [{"event": "identity", "time_unix": self.base + delta, "monotonic_ns": 1000000000 + int((delta + 1) * 1000000000),
                       "pid": 12345, "start_ticks": 999, "pgid": 12345, "cgroup": "0::" + self.cg + "\n",
                       "executable_verified": True, "argv_verified": True, "environment_verified": True,
                       "binary_sha256": "a" * 64, "seccomp_filters": 1 if index < 3 else 2,
                       "terminal_exec_filter_verified": index == 3, **({"completion_verified": True} if index == 3 else {})}
                      for index, delta in enumerate((-0.1, 0.25, 0.75, 1.1))]
        identities.append({"event": "cleanup", "time_unix": self.base + 1.2, "monotonic_ns": 3200000000, "live_process_group_after": []})
        self.write_rows(identities)
        write_json(self.root / "identity/summary.json", {"verdict": "PASS", "qualification": "Python_probe_identity_only", "identity_samples": 4,
                   "probe_exit_code": 0, "live_process_group_after": [], "failure": None, "raw_sha256": digest(self.root / "identity/raw.jsonl")})
        samples = []
        for delta in (0, 0.5, 1):
            stamp = datetime.fromtimestamp(self.base + delta, timezone.utc).isoformat()
            samples.append(f"{stamp} mem_avail_kb=120000000 eng_rss_kb=100000 read_bytes=0 cgroup_current_bytes=200000000 cgroup_peak_bytes=300000000 cgroup_swap_current_bytes=0\n")
        (self.root / "samples.log").write_text("".join(samples))
        (self.root / "kernel.log").write_text("-- No entries --\n")
        (self.root / "main.log").write_text(
            "2023-11-14T22:13:17.900000+00:00 SAFE_RUN start tag=glm53-native-smoke-004 vlimit_kb=419430400 kill_floor_gib=40 min_start_gib=110 timeout_s=600 allow_cgroup_high=0\n"
            f"2023-11-14T22:13:18+00:00 cgroup_verified path={self.cg} memory_high=68719476736 memory_max=73014444032 memory_swap_max=0 memory_oom_group=1\n"
            'MemTotal: 125483612 kB\nMemAvailable: 120000000 kB\n'
            '{"pass":true,"required_gib":110.0,"mem_available_gib":114.441,"stable_samples_observed":3}\n'
            '2023-11-14T22:13:19+00:00 wrapper_pid=54321 engine_pid=12345 pgid=54321 (periodic sampler)\n'
            '2023-11-14T22:13:22+00:00 cgroup_final current_bytes=0 peak_bytes=300000000 swap_current_bytes=0 events=low 0,high 0,max 0,oom 0,oom_kill 0,oom_group_kill 0,\n'
            '2023-11-14T22:13:22+00:00 SAFE_RUN end rc=0 killed=no (124=timeout, 137=SIGKILL/ENOMEM-adjacent)\n')
        live = {"Id": self.unit, "LoadState": "loaded", "ActiveState": "active", "SubState": "running", "MainPID": "55555", "ControlGroup": self.cg,
                "MemoryHigh": "68719476736", "MemoryMax": "73014444032", "MemorySwapMax": "0", "OOMPolicy": "kill", "KillMode": "control-group"}
        after = {**live, "LoadState": "not-found", "ActiveState": "inactive", "SubState": "dead", "MainPID": "0", "ControlGroup": ""}
        for name, values, delta in (("unit-live.json", live, 0.05), ("unit-after.json", after, 2.1)):
            write_json(self.root / name, {"command": unit_query(self.unit), "returncode": 0, "observed_at": self.base + delta,
                       "stdout": "".join(f"{k}={v}\n" for k, v in values.items()), "stderr": ""})
        write_json(self.root / "cgroup-after.json", {"path": "/sys/fs/cgroup" + self.cg, "exists": False, "observed_at": self.base + 2.2})
        for name, delta in (("swap-before.json", -3), ("swap-after.json", 2.3)):
            write_json(self.root / name, {"path": "/proc/vmstat", "observed_at": self.base + delta, "text": "pswpin 10\npswpout 20\n"})
        self.seal()

    def write_rows(self, rows):
        (self.root / "identity/raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def seal(self):
        path = self.root / "identity/summary.json"
        if path.exists():
            summary = json.loads(path.read_text()); summary["raw_sha256"] = digest(self.root / "identity/raw.jsonl"); write_json(path, summary)
        (self.root / "wrapper.log").write_text(f"SAFE_RUN_DONE rc=0 killed=no dir=/raw/crash main_sha256={digest(self.root / 'main.log')} samples_sha256={digest(self.root / 'samples.log')} kernel_sha256={digest(self.root / 'kernel.log')}\n")

    def test_stale_wrapper_chronology_rejects(self):
        path = self.root / "main.log"
        path.write_text(path.read_text().replace("2023-11-14", "2000-11-14")); self.seal()
        with self.assertRaisesRegex(ValueError, "chronology|window|timestamp"):
            score_host_observations(self.root, self.expected)

    def test_individual_wrapper_events_must_fit_probe_window(self):
        path = self.root / "main.log"; original = path.read_text()
        for event in ("SAFE_RUN start", "cgroup_verified", "wrapper_pid=", "cgroup_final", "SAFE_RUN end"):
            with self.subTest(event=event):
                path.write_text("\n".join(line.replace("2023-11-14", "2024-11-14") if event in line else line
                                          for line in original.splitlines()) + "\n"); self.seal()
                with self.assertRaisesRegex(ValueError, "chronology|cleanup"):
                    score_host_observations(self.root, self.expected)

    def test_wrapper_process_discovery_can_follow_initial_identity(self):
        path = self.root / "main.log"
        path.write_text(path.read_text().replace("22:13:19+00:00 wrapper_pid", "22:13:19.950000+00:00 wrapper_pid")); self.seal()
        self.assertEqual(score_host_observations(self.root, self.expected)["verdict"], "PASS")

    def test_missing_or_changed_frozen_start_controls_reject(self):
        path = self.root / "main.log"; original = path.read_text()
        variants = ["\n".join(original.splitlines()[1:]) + "\n"]
        variants += [original.replace(old, new) for old, new in (
            ("kill_floor_gib=40", "kill_floor_gib=0"), ("min_start_gib=110", "min_start_gib=1"),
            ("timeout_s=600", "timeout_s=601"), ("allow_cgroup_high=0", "allow_cgroup_high=1"),
            ("tag=glm53-native-smoke-004", "tag=glm53-native-smoke-003"))]
        for text in variants:
            with self.subTest(text=text.splitlines()[0]):
                path.write_text(text); self.seal()
                with self.assertRaisesRegex(ValueError, "start|controls"):
                    score_host_observations(self.root, self.expected)

    def test_contradictory_or_malformed_terminal_records_reject(self):
        for name, failure in (("wrapper.log", "SAFE_RUN_DONE rc=124 killed=timeout dir=/failed\n"),
                              ("main.log", "2023-11-14T22:13:22+00:00 SAFE_RUN end rc=124 killed=timeout (124=timeout, 137=SIGKILL/ENOMEM-adjacent)\n"),
                              ("wrapper.log", "SAFE_RUN_DONE malformed\n"),
                              ("main.log", "2023-11-14T22:13:22+00:00 SAFE_RUN end malformed\n")):
            with self.subTest(name=name, failure=failure):
                path = self.root / name; original = path.read_text()
                path.write_text(original + failure)
                if name == "main.log": self.seal()
                with self.assertRaisesRegex(ValueError, "completion|terminal|exit"):
                    score_host_observations(self.root, self.expected)
                path.write_text(original); self.seal()

    def test_complete_synthetic_bundle_scores_only_host_scope(self):
        result = score_host_observations(self.root, self.expected)
        self.assertEqual(result["qualification"], "host_and_probe_identity_observations_only")
        self.assertEqual(result["minimum_mem_available_kib"], 120000000)
        self.assertEqual(result["identity_samples"], 4)

    def test_low_memory_and_swap_reject_despite_valid_hashes(self):
        path = self.root / "samples.log"; original = path.read_text()
        for old, new, message in (("120000000", "100", "memory"), ("cgroup_swap_current_bytes=0", "cgroup_swap_current_bytes=1", "swap")):
            with self.subTest(message=message):
                path.write_text(original.replace(old, new)); self.seal()
                with self.assertRaisesRegex(ValueError, message): score_host_observations(self.root, self.expected)
        path.write_text(original)

    def test_kernel_fault_and_incomplete_journal_reject(self):
        for text in ("NVRM: Xid (PCI:0000:01:00): 31\n", "", "Failed to read journal\n"):
            (self.root / "kernel.log").write_text(text); self.seal()
            with self.assertRaisesRegex(ValueError, "kernel"): score_host_observations(self.root, self.expected)

    def test_missing_or_false_final_identity_rejects(self):
        path = self.root / "identity/raw.jsonl"; original = [json.loads(line) for line in path.read_text().splitlines()]
        for key in ("completion_verified", "terminal_exec_filter_verified", "argv_verified"):
            rows = json.loads(json.dumps(original)); rows[-2][key] = False; self.write_rows(rows); self.seal()
            with self.assertRaisesRegex(ValueError, "identity|completion|filter"): score_host_observations(self.root, self.expected)

    def test_identity_pid_and_binary_substitution_reject(self):
        path = self.root / "identity/raw.jsonl"; original = [json.loads(line) for line in path.read_text().splitlines()]
        for key, value in (("pid", 23456), ("start_ticks", 1000), ("binary_sha256", "e" * 64)):
            rows = json.loads(json.dumps(original)); rows[1][key] = value; self.write_rows(rows); self.seal()
            with self.assertRaisesRegex(ValueError, "identity"): score_host_observations(self.root, self.expected)

    def test_counterfeit_cleanup_or_containment_rejects(self):
        path = self.root / "unit-after.json"; value = json.loads(path.read_text()); value["stdout"] = value["stdout"].replace("MainPID=0", "MainPID=55555"); write_json(path, value)
        with self.assertRaisesRegex(ValueError, "cleanup"): score_host_observations(self.root, self.expected)

    def test_actual_unit_must_enforce_group_kill_and_zero_swap(self):
        path = self.root / "unit-live.json"; original = json.loads(path.read_text())
        for old, new in (("OOMPolicy=kill", "OOMPolicy=continue"), ("KillMode=control-group", "KillMode=process"), ("MemorySwapMax=0", "MemorySwapMax=4096")):
            value = dict(original); value["stdout"] = value["stdout"].replace(old, new); write_json(path, value)
            with self.assertRaisesRegex(ValueError, "containment"): score_host_observations(self.root, self.expected)

    def test_sampling_gap_and_missing_rows_reject(self):
        path = self.root / "samples.log"; original = path.read_text()
        for text in (original.splitlines()[0] + "\n", original.replace("22:13:20.500000", "22:13:25.500000")):
            path.write_text(text); self.seal()
            with self.assertRaisesRegex(ValueError, "coverage"): score_host_observations(self.root, self.expected)

    def test_cgroup_path_must_be_gone_after_exit(self):
        path = self.root / "cgroup-after.json"; value = json.loads(path.read_text()); value["exists"] = True; write_json(path, value)
        with self.assertRaisesRegex(ValueError, "cleanup"): score_host_observations(self.root, self.expected)

    def test_probe_seed_and_environment_remain_frozen(self):
        path = self.root / "identity/manifest.json"; original = json.loads(path.read_text())
        for key, value in (("argv", original["argv"][:-1] + ["2"]), ("environment_sha256", "e" * 64)):
            changed = dict(original); changed[key] = value; write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "identity"): score_host_observations(self.root, self.expected)

    def test_whole_system_swap_delta_rejects(self):
        path = self.root / "swap-after.json"; value = json.loads(path.read_text()); value["text"] = "pswpin 10\npswpout 21\n"; write_json(path, value)
        with self.assertRaisesRegex(ValueError, "swap"): score_host_observations(self.root, self.expected)

    def test_tampered_unbound_raw_log_rejects(self):
        with (self.root / "samples.log").open("a") as stream: stream.write("changed\n")
        with self.assertRaisesRegex(ValueError, "digest"): score_host_observations(self.root, self.expected)

    def test_duplicate_and_nonfinite_identity_json_rejects(self):
        path = self.root / "identity/raw.jsonl"; original = path.read_text()
        for replacement in ('"pid": 12345, "pid": 12345', '"pid": NaN'):
            path.write_text(original.replace('"pid": 12345', replacement, 1)); self.seal()
            with self.assertRaises(ValueError): score_host_observations(self.root, self.expected)


if __name__ == "__main__":
    unittest.main()
