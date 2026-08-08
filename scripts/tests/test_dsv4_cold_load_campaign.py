#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = load_module("dsv4_cold_scorer", ROOT / "scripts/94_score_dsv4_cold_load.py")
RUNNER = load_module("dsv4_cold_runner", ROOT / "scripts/95_run_dsv4_cold_load.py")
MODEL_BYTES = 96_832_507_552
SHA = {name: char * 64 for name, char in {
    "candidate": "1", "runner": "2", "scorer": "3", "model": "4",
    "config": "5", "binary": "6", "semantic": "7", "logit": "8",
    "randomness": "9", "receipt": "a", "closure": "b",
    "drand_verifier": "c", "drand_node": "d", "runtime_bundle": "e",
}.items()}


def expected_schedules(randomness_hex: str) -> list[str]:
    seed = bytes.fromhex(randomness_hex)
    domain = b"frontier-at-home/dsv4-cold-load/v1\0"
    return [
        "ABBA" if hashlib.sha256(domain + seed + bytes([block])).digest()[0] & 1 == 0 else "BAAB"
        for block in range(5)
    ]


def manifest() -> dict[str, object]:
    randomness = SHA["randomness"]
    return {
        "schema_version": 1,
        "candidate_hash": SHA["candidate"],
        "runner_sha256": SHA["runner"],
        "scorer_sha256": SHA["scorer"],
        "model_sha256": SHA["model"],
        "configuration_sha256": SHA["config"],
        "binary_sha256": SHA["binary"],
        "runtime_bundle_sha256": SHA["runtime_bundle"],
        "drand_verifier_sha256": SHA["drand_verifier"],
        "drand_node_sha256": SHA["drand_node"],
        "model_bytes": MODEL_BYTES,
        "randomness": {
            "value": randomness,
            "receipt_sha256": SHA["receipt"],
        },
        "schedules": expected_schedules(randomness),
    }


def rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    schedules = expected_schedules(SHA["randomness"])
    for block, schedule in enumerate(schedules):
        for position, letter in enumerate(schedule):
            arm = "off" if letter == "A" else "on"
            launch = (block * 4 + position + 1) * 100_000_000_000
            ready_seconds = 90.0 if arm == "off" else 18.0
            tensor_seconds = 80.0 if arm == "off" else 16.0
            result.append({
                "schema_version": 1,
                "block": block,
                "position": position,
                "arm": arm,
                "run_id": f"b{block}-p{position}",
                "candidate_hash": SHA["candidate"],
                "model_sha256": SHA["model"],
                "configuration_sha256": SHA["config"],
                "binary_sha256": SHA["binary"],
                "runtime_closure_sha256": SHA["closure"],
                "runtime_closure_count": 12,
                "process_launch_monotonic_ns": launch,
                "health_ready_monotonic_ns": launch + int(ready_seconds * 1e9),
                "tensor_load_start_monotonic_ns": launch + 1_000_000_000,
                "tensor_load_end_monotonic_ns": launch + 1_000_000_000 + int(tensor_seconds * 1e9),
                "server_pid": 10_000 + block * 4 + position,
                "server_start_ticks": 20_000 + block * 4 + position,
                "server_fresh": True,
                "physical_read_bytes": MODEL_BYTES,
                "cache_resident_bytes_before": 0,
                "direct_shard_count": 3 if arm == "on" else 0,
                "direct_required": arm == "on",
                "semantic_sha256": SHA["semantic"],
                "first_token_logit_sha256": SHA["logit"],
                "authenticated_health": True,
                "authenticated_completion": True,
                "unauthenticated_rejected": True,
                "minimum_mem_available_kb": 50 * 1024 * 1024,
                "swap_growth_bytes": 0,
                "cgroup_oom_delta": 0,
                "cgroup_oom_kill_delta": 0,
                "cgroup_max_delta": 0,
                "xid_count": 0,
                "surviving_descendants": 0,
                "systemd_result": "success",
                "systemd_exec_main_code": 0,
                "systemd_exec_main_status": 0,
                "systemd_memory_peak_bytes": 80 * 1024**3,
                "systemd_memory_swap_peak_bytes": 0,
            })
    return result


class Dsv4ColdLoadCampaignTests(unittest.TestCase):
    def test_runtime_bundle_digest_binds_files_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            artifact = bundle / "llama-server"
            artifact.write_bytes(b"first")
            artifact.chmod(0o755)
            os.symlink("llama-server", bundle / "server-link")
            first = RUNNER.runtime_bundle_sha256(bundle)
            artifact.write_bytes(b"second")
            self.assertNotEqual(RUNNER.runtime_bundle_sha256(bundle), first)
            os.unlink(bundle / "server-link")
            os.symlink("/etc/passwd", bundle / "server-link")
            with self.assertRaisesRegex(RUNNER.CampaignError, "runtime bundle"):
                RUNNER.runtime_bundle_sha256(bundle)

    def test_randomness_derives_five_balanced_blocks(self) -> None:
        schedules = RUNNER.arm_schedule(SHA["randomness"])
        self.assertEqual(schedules, expected_schedules(SHA["randomness"]))
        self.assertEqual(RUNNER.arm_schedule(SHA["randomness"]), schedules)
        self.assertNotEqual(RUNNER.arm_schedule("b" * 64), schedules)

    def test_runner_materializes_twenty_unique_fresh_arm_plans(self) -> None:
        plans = RUNNER.campaign_plan(SHA["randomness"])
        self.assertEqual(len(plans), 20)
        self.assertEqual(len({plan.run_id for plan in plans}), 20)
        for plan in plans:
            expected = expected_schedules(SHA["randomness"])[plan.block][plan.position]
            self.assertEqual(plan.arm, "off" if expected == "A" else "on")
            self.assertEqual(
                RUNNER.direct_io_arguments(plan.arm),
                ["--direct-io-required"] if plan.arm == "on" else ["--no-direct-io"],
            )

    def test_runner_requires_working_containment_and_idle_host(self) -> None:
        with mock.patch.object(RUNNER.subprocess, "run") as completed:
            completed.side_effect = [
                mock.Mock(returncode=1, stdout="", stderr="manager dead"),
            ]
            with self.assertRaisesRegex(RUNNER.CampaignError, "user-systemd containment"):
                RUNNER.preflight_host()

        def fake_run(command, **_kwargs):
            if command[-2:] == ["-x", "ds4-server"]:
                return mock.Mock(returncode=0, stdout="123\n", stderr="")
            return mock.Mock(returncode=0, stdout="running\n", stderr="")

        with mock.patch.object(RUNNER.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RUNNER.CampaignError, "engine or fio"):
                RUNNER.preflight_host()

    def test_runner_observes_exact_shard_descriptors_and_direct_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary) / "proc"
            fd = proc / "321" / "fd"
            fdinfo = proc / "321" / "fdinfo"
            fd.mkdir(parents=True)
            fdinfo.mkdir()
            shards = [Path(temporary) / f"model-0000{i}-of-00003.gguf" for i in range(1, 4)]
            for index, shard in enumerate(shards, start=7):
                shard.touch()
                os.symlink(shard, fd / str(index))
                (fdinfo / str(index)).write_text(
                    f"pos:\t0\nflags:\t{os.O_RDONLY | os.O_DIRECT:o}\n", encoding="ascii"
                )
            self.assertEqual(RUNNER.direct_shard_count(321, shards, proc_root=proc), 3)
            (fdinfo / "8").write_text(f"pos:\t0\nflags:\t{os.O_RDONLY:o}\n", encoding="ascii")
            self.assertEqual(RUNNER.direct_shard_count(321, shards, proc_root=proc), 2)

    def test_runner_bounds_each_arm_and_never_reuses_output(self) -> None:
        command = RUNNER.containment_command(
            "cold-b0-p0-abcdefabcdef", ["/frozen/llama-server", "--help"], Path("/tmp/run.log")
        )
        joined = " ".join(command)
        self.assertIn("MemoryHigh=100G", joined)
        self.assertIn("MemoryMax=104G", joined)
        self.assertIn("MemorySwapMax=0", joined)
        self.assertIn("OOMPolicy=kill", joined)
        self.assertIn("KillMode=control-group", joined)
        self.assertIn("RuntimeMaxSec=300s", joined)
        self.assertIn("WorkingDirectory=/", joined)
        self.assertNotIn("--collect", command)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "attempt"
            destination.mkdir()
            (destination / "old").write_text("occupied", encoding="ascii")
            with self.assertRaisesRegex(RUNNER.CampaignError, "fresh empty"):
                RUNNER.require_fresh_output(destination)

    def test_accepts_matched_safe_fast_campaign(self) -> None:
        result = SCORER.score_campaign(manifest(), rows())
        self.assertEqual(result["verdict"], "PASS")
        self.assertLessEqual(result["observed"]["ready_ratio_upper_95"], 0.5)
        self.assertLessEqual(result["observed"]["on_tensor_seconds_upper_95"], 20.393206603359758)

    def test_rejects_incomplete_duplicate_unmatched_or_stale_rows(self) -> None:
        mutations = []
        bad = rows()[:-1]; mutations.append(bad)
        bad = rows(); bad[1]["run_id"] = bad[0]["run_id"]; mutations.append(bad)
        bad = rows(); bad[2]["model_sha256"] = "f" * 64; mutations.append(bad)
        bad = rows(); bad[3]["binary_sha256"] = "e" * 64; mutations.append(bad)
        bad = rows(); bad[4]["arm"] = "on" if bad[4]["arm"] == "off" else "off"; mutations.append(bad)
        for raw in mutations:
            self.assertEqual(SCORER.score_campaign(manifest(), raw)["verdict"], "FAIL")

    def test_runner_rejects_unsigned_randomness_and_observes_loaded_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "randomness.json"
            receipt.write_text(
                json.dumps({"randomness": SHA["randomness"]}), encoding="utf-8"
            )
            with self.assertRaisesRegex(RUNNER.CampaignError, "randomness"):
                RUNNER._load_randomness(receipt)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc = root / "proc" / "321"
            proc.mkdir(parents=True)
            executable = root / "llama-server"
            library = root / "libllama.so"
            unrelated = root / "CMakeCache.txt"
            executable.write_bytes(b"server")
            library.write_bytes(b"library")
            unrelated.write_bytes(b"first")
            executable.chmod(0o755)
            library.chmod(0o755)
            os.symlink(executable, proc / "exe")
            library_stat = library.stat()
            (proc / "maps").write_text(
                f"1000-2000 r-xp 00000000 {os.major(library_stat.st_dev):x}:"
                f"{os.minor(library_stat.st_dev):x} {library_stat.st_ino} {library}\n"
                f"2000-3000 rw-p 00000000 00:00 2 {unrelated}\n",
                encoding="utf-8",
            )
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            first_digest, first_count = RUNNER.capture_runtime_closure(
                321, first, proc_root=root / "proc"
            )
            unrelated.write_bytes(b"changed but never mapped")
            second_digest, second_count = RUNNER.capture_runtime_closure(
                321, second, proc_root=root / "proc"
            )
            self.assertEqual((first_digest, first_count), (second_digest, second_count))
            self.assertEqual(first_count, 2)
            (proc / "maps").write_text(
                f"1000-2000 r-xp 00000000 {os.major(library_stat.st_dev):x}:"
                f"{os.minor(library_stat.st_dev):x} {library_stat.st_ino + 1} {library}\n",
                encoding="utf-8",
            )
            third = root / "third"
            third.mkdir()
            with self.assertRaisesRegex(RUNNER.CampaignError, "mapped runtime identity"):
                RUNNER.capture_runtime_closure(321, third, proc_root=root / "proc")
        for field in (
            "ExecMainCode", "ExecMainStatus", "Result", "MemoryPeak", "MemorySwapPeak",
        ):
            self.assertIn(
                field,
                (ROOT / "scripts/95_run_dsv4_cold_load.py").read_text(encoding="utf-8"),
            )

    def test_randomness_bls_and_systemd_properties_fail_closed(self) -> None:
        receipt = {
            "round": RUNNER.DRAND_FREEZE_FLOOR_ROUND + 1,
            "freeze_floor_round": RUNNER.DRAND_FREEZE_FLOOR_ROUND,
            "randomness": SHA["randomness"],
            "signature": "1" * 192,
            "previous_signature": "2" * 192,
            "frozen_gate_commit": SHA["candidate"],
            "relay_agreement": ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            digest = lambda path: (
                RUNNER.DRAND_VERIFIER_SHA256
                if path == RUNNER.DRAND_VERIFIER
                else RUNNER.DRAND_NODE_SHA256
            )
            with mock.patch.object(RUNNER, "_sha256", side_effect=digest), mock.patch.object(
                RUNNER, "_run", return_value=mock.Mock(
                    returncode=0, stdout="DRAND_BLS_RECEIPT_OK\n", stderr=""
                )
            ):
                randomness, _, raw = RUNNER._load_randomness(path, SHA["candidate"])
                self.assertEqual(randomness, SHA["randomness"])
                self.assertEqual(raw, path.read_bytes())
            receipt["frozen_gate_commit"] = "f" * 64
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.CampaignError, "candidate-bound"):
                RUNNER._load_randomness(path, SHA["candidate"])

        good_properties = "\n".join([
            "ExecMainCode=0", "ExecMainStatus=0", "Result=success",
            "MemoryPeak=1234", "MemorySwapPeak=0", "",
        ])
        with mock.patch.object(
            RUNNER, "_run", return_value=mock.Mock(
                returncode=0, stdout=good_properties, stderr=""
            )
        ):
            self.assertEqual(RUNNER._unit_properties("cold-b0-p0-abcdefabcdef")["Result"], "success")
        with mock.patch.object(
            RUNNER, "_run", return_value=mock.Mock(
                returncode=0, stdout=good_properties.replace("Result=success", "Result=oom-kill"), stderr=""
            )
        ):
            with self.assertRaisesRegex(RUNNER.CampaignError, "failure"):
                RUNNER._unit_properties("cold-b0-p0-abcdefabcdef")

    def test_rejects_warm_fallback_unsafe_or_unobserved_arms(self) -> None:
        fields = {
            "cache_resident_bytes_before": 2 * 1024**3,
            "physical_read_bytes": MODEL_BYTES // 2,
            "server_fresh": False,
            "minimum_mem_available_kb": 9 * 1024 * 1024,
            "swap_growth_bytes": 4096,
            "cgroup_oom_kill_delta": 1,
            "cgroup_max_delta": 1,
            "xid_count": 1,
            "surviving_descendants": 1,
            "systemd_result": "oom-kill",
            "systemd_memory_swap_peak_bytes": 4096,
        }
        for key, value in fields.items():
            bad = rows(); bad[1][key] = value
            self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL", key)
        bad = rows(); on = next(row for row in bad if row["arm"] == "on"); on["direct_shard_count"] = 2
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); off = next(row for row in bad if row["arm"] == "off"); off["direct_shard_count"] = 1
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")

    def test_rejects_auth_semantic_and_timing_failures(self) -> None:
        for key in ("authenticated_health", "authenticated_completion", "unauthenticated_rejected"):
            bad = rows(); bad[0][key] = False
            self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["semantic_sha256"] = "c" * 64
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["first_token_logit_sha256"] = "d" * 64
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["runtime_closure_sha256"] = "e" * 64
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[1]["health_ready_monotonic_ns"] = bad[1]["process_launch_monotonic_ns"]
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")

    def test_rejects_performance_regression_and_nonfinite_values(self) -> None:
        bad = rows()
        for row in bad:
            if row["arm"] == "on":
                row["health_ready_monotonic_ns"] = row["process_launch_monotonic_ns"] + 60_000_000_000
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")
        bad = rows(); bad[0]["physical_read_bytes"] = float("nan")
        self.assertEqual(SCORER.score_campaign(manifest(), bad)["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
