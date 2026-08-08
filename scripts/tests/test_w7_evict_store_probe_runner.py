#!/usr/bin/env python3

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/93_run_w7_evict_store_probe.py"
SPEC = importlib.util.spec_from_file_location("w7_evict_store_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class W7EvictStoreProbeRunnerTests(unittest.TestCase):
    def terminal_row(self, arm: str, position: int, config_sha: str, request_sha: str) -> dict[str, object]:
        tokens = list(range(128))
        first = 2_000_000_000 if arm == "off" else 1_000_000_000
        timestamps = [first + index * 100_000_000 for index in range(128)]
        return {
            "arm": arm,
            "position": position,
            "run_id": f"run-{position}-{arm}",
            "binary_sha256": MODULE.BINARY_SHA256,
            "model_sha256": MODULE.MODEL_SHA256,
            "common_config_sha256": config_sha,
            "request_sha256": request_sha,
            "diagnostic_skip": 1 if arm == "on" else 0,
            "request_start_ns": 0,
            "token_timestamps_ns": timestamps,
            "output_token_ids": tokens,
            "output_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
            "generated_text_sha256": hashlib.sha256(b"same output").hexdigest(),
            "generated_text_bytes": len(b"same output"),
            "logit_sha256s": [hashlib.sha256(f"logit-{index}".encode()).hexdigest() for index in range(3)],
            "selected_checkpoint_tokens": 5044,
            "checkpoint_id": "token-text:" + "9" * 40,
            "evict_store_count": 0 if arm == "on" else 1,
            "skip_marker_count": 1 if arm == "on" else 0,
            "activation_marker_count": 1 if arm == "on" else 0,
            "server_fresh": True,
            "safety": {
                "containment_rc": 0,
                "minimum_mem_available_kb": 48_000_000,
                "swap_growth_bytes": 0,
                "cgroup_max_delta": 0,
                "cgroup_oom_delta": 0,
                "cgroup_oom_kill_delta": 0,
                "xid_count": 0,
                "surviving_descendants": 0,
            },
            "executed_environment_sha256": hashlib.sha256(f"env-{arm}".encode()).hexdigest(),
        }

    def make_normal_terminal(self, attempt: Path) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
        seed = "a" * 64
        order = MODULE.derive_order(seed)
        config = {
            "binary_sha256": MODULE.BINARY_SHA256,
            "model_sha256": MODULE.MODEL_SHA256,
            "context": 8192,
        }
        config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        request_sha = "4" * 64
        rows = [self.terminal_row(arm, position, config_sha, request_sha) for position, arm in enumerate(order)]
        scorer_rows = []
        for row in rows:
            scorer_row = dict(row)
            scorer_row.pop("executed_environment_sha256")
            scorer_rows.append(scorer_row)
        summary = MODULE.load_scorer(MODULE.SCORER.read_bytes()).score_probe_rows(scorer_rows, order)
        raw = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in rows
        )
        summary_bytes = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (attempt / "raw.jsonl").write_bytes(raw)
        (attempt / "summary.json").write_bytes(summary_bytes)
        manifest = {
            "schema": "glm52-w7-evict-store-probe-v1",
            "candidate_hash": "c" * 40,
            "verdict": summary["verdict"],
            "completed_rows": len(rows),
            "runner_sha256": hashlib.sha256(MODULE.Path(MODULE.__file__).read_bytes()).hexdigest(),
            "base_lifecycle_sha256": MODULE.BASE_SHA256,
            "scorer_sha256": MODULE.SCORER_SHA256,
            "cgroup_sha256": MODULE.CGROUP_SHA256,
            "safe_run_sha256": MODULE.SAFE_SHA256,
            "memory_guard_sha256": MODULE.MEMORY_GUARD_SHA256,
            "binary_sha256": MODULE.BINARY_SHA256,
            "engine_source_commit": MODULE.ENGINE_SOURCE_COMMIT,
            "model_sha256": MODULE.MODEL_SHA256,
            "model_bytes": MODULE.MODEL_BYTES,
            "live_request_sha256": MODULE.LIVE_SHA256,
            "primary_source_sha256": MODULE.PRIMARY_SHA256,
            "executed_request_sha256": request_sha,
            "configuration": config,
            "configuration_sha256": config_sha,
            "public_randomness_sha256": seed,
            "public_randomness_receipt_sha256": "7" * 64,
            "arm_order": order,
            "artifacts": {
                "raw.jsonl": hashlib.sha256(raw).hexdigest(),
                "summary.json": hashlib.sha256(summary_bytes).hexdigest(),
            },
        }
        (attempt / "manifest.json").write_text(json.dumps(manifest))
        return manifest, rows, summary

    def rewrite_terminal(self, attempt: Path, manifest: dict[str, object], rows: list[dict[str, object]], summary: dict[str, object]) -> None:
        raw = b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in rows
        )
        summary_bytes = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
        (attempt / "raw.jsonl").write_bytes(raw)
        (attempt / "summary.json").write_bytes(summary_bytes)
        manifest["completed_rows"] = len(rows)
        manifest["verdict"] = summary["verdict"]
        manifest["artifacts"] = {
            "raw.jsonl": hashlib.sha256(raw).hexdigest(),
            "summary.json": hashlib.sha256(summary_bytes).hexdigest(),
        }
        (attempt / "manifest.json").write_text(json.dumps(manifest))

    def bind_server(self, out: Path) -> str:
        server = out / "server.log"
        metadata = server.stat()
        digest = hashlib.sha256(server.read_bytes()).hexdigest()
        safety = out / "safety"
        safety.mkdir(exist_ok=True)
        main = (
            f"final_artifact_verified path={server} sha256={digest} "
            f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}\n"
        ).encode()
        (safety / "main.log").write_bytes(main)
        return (
            "SAFE_RUN_DONE rc=0 killed=no "
            "dir=/home/bmarti44/.local/state/glm52-crashlog/unit-test "
            f"main_sha256={hashlib.sha256(main).hexdigest()} "
            f"samples_sha256={'a' * 64} kernel_sha256={'b' * 64}\n"
        )

    def make_parse_arm(self) -> tuple[tempfile.TemporaryDirectory, Path, dict[str, object], str]:
        temporary = tempfile.TemporaryDirectory(prefix="w7-evict-parse-")
        out = Path(temporary.name)
        server = (
            "ds4-server: listening on http://127.0.0.1:8097\n"
            "ds4-server: kv cache stored tokens=5055 trimmed=0 reason=evict key=token-text size=918.82 MiB save=700.0 ms\n"
            "ds4-server: kv cache hit text tokens=5044 text=15571 quant=2 key=token-text load=500.0 ms file="
            + str(out / "kv/9e5ba8aa0b75e6c618f68d9834ef541c44cd4b42.kv") + "\n"
            "ds4-server: shutdown requested\n"
        )
        (out / "server.log").write_text(server)
        records = []
        for index, start, prompt, suffix in (
            (1, 0, 5044, 5044), (2, 5044, 5055, 11), (3, 5044, 5066, 22),
        ):
            path = out / f"logits.sync{index}.start{start}.prompt{prompt}.suffix{suffix}"
            path.write_bytes(index.to_bytes(4, "little"))
            records.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest(), 4))
        sequence = hashlib.sha256(json.dumps(records, separators=(",", ":")).encode("ascii")).hexdigest()
        base_row = {
            "block": 0, "position": 0, "arm": "off", "run_id": "run-off",
            "binary_sha256": MODULE.BINARY_SHA256, "model_sha256": MODULE.MODEL_SHA256,
            "common_config_sha256": "3" * 64, "request_sha256": "4" * 64,
            "stable_remap": 0, "request_start_ns": 1,
            "token_timestamps_ns": list(range(2, 130)), "output_token_ids": list(range(128)),
            "output_sha256": "5" * 64, "generated_text_sha256": "6" * 64,
            "generated_text_bytes": 1, "final_logits_sha256": records[-1][1],
            "logit_sequence_sha256": sequence, "server_fresh": True,
            "safety": {
                "containment_rc": 0, "minimum_mem_available_kb": 48_000_000,
                "swap_growth_bytes": 0, "cgroup_max_delta": 0, "cgroup_oom_delta": 0,
                "cgroup_oom_kill_delta": 0, "xid_count": 0, "surviving_descendants": 0,
                "false_generation_flushes": 300,
            },
        }
        return temporary, out, base_row, self.bind_server(out)

    def convert_server_to_on(self, out: Path) -> str:
        path = out / "server.log"
        source = path.read_text()
        store = "ds4-server: kv cache stored tokens=5055 trimmed=0 reason=evict key=token-text size=918.82 MiB save=700.0 ms\n"
        source = source.replace(store, MODULE.ACTIVATION + "\n" + MODULE.SKIPPED + "\n")
        path.write_text(source)
        return self.bind_server(out)

    def test_self_test_checks_dependencies_without_engine(self) -> None:
        before = subprocess.run(["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True).stdout
        completed = subprocess.run(
            ["/usr/bin/python3", str(RUNNER), "--self-test"],
            capture_output=True, text=True, timeout=30,
        )
        after = subprocess.run(["/usr/bin/pgrep", "-x", "ds4-server"], capture_output=True, text=True).stdout
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout, "W7_EVICT_STORE_PROBE_SELFTEST_OK\n")
        self.assertEqual(after, before)

    def test_order_is_deterministic_and_balanced(self) -> None:
        for seed in ("a" * 64, "b" * 64, "0" * 64, "f" * 64):
            order = MODULE.derive_order(seed)
            self.assertEqual(order, MODULE.derive_order(seed))
            self.assertEqual(set(order), {"off", "on"})
        for invalid in ("", "a" * 63, "A" * 64, "z" * 64):
            with self.assertRaises(ValueError):
                MODULE.derive_order(invalid)

    def test_runner_uses_frozen_lifecycle_and_bounded_flag(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for required in (
            "91_run_w7_cache_generation_campaign.py", "glm_safe_run.sh",
            "glm_cgroup_run_w7_evict_store_v1.sh", MODULE.FLAG,
            "MemorySwapMax", "minimum_start_GiB", "model content identity mismatch",
            "evict_store_count", "selected_checkpoint_tokens", "logit_sha256s",
            "manifest.json", "raw.jsonl", "summary.json", "public_randomness",
            "install_campaign_signal_handlers", "finalize_failure_triplet",
            'manifest["verdict"] = verdict',
        ):
            self.assertIn(required, source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("sudo", source)
        self.assertNotIn("reboot", source)

    def test_post_attempt_failure_is_preserved_as_triplet(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-evict-finalize-") as temporary:
            attempt = Path(temporary)
            MODULE._ACTIVE_ATTEMPT = attempt
            MODULE._ACTIVE_CANDIDATE = "a" * 40
            MODULE._ACTIVE_ROWS = [{"arm": "off", "run_id": "completed"}]
            arm = attempt / "p0-off"
            arm.mkdir()
            (arm / "server.log").write_text("bound arm evidence\n")
            try:
                MODULE.finalize_failure_triplet(RuntimeError("injected"))
            finally:
                MODULE._ACTIVE_ATTEMPT = None
                MODULE._ACTIVE_CANDIDATE = None
                MODULE._ACTIVE_ROWS = []
            self.assertIn('"run_id":"completed"', (attempt / "raw.jsonl").read_text())
            summary = json.loads((attempt / "summary.json").read_text())
            manifest = json.loads((attempt / "manifest.json").read_text())
            self.assertEqual(summary["verdict"], "FAIL")
            self.assertIn("RuntimeError: injected", summary["failure"])
            self.assertEqual(manifest["schema"], "glm52-w7-evict-store-probe-failure-v1")
            self.assertEqual(manifest["candidate_hash"], "a" * 40)
            self.assertEqual(manifest["verdict"], "FAIL")
            self.assertIn("p0-off/server.log", manifest["artifacts"])

    def test_finalizer_recovers_base_activation_window_and_partial_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-evict-window-") as temporary:
            attempt = Path(temporary)
            (attempt / "manifest.json").write_text("{partial")
            MODULE._ACTIVE_ATTEMPT = None
            MODULE._ACTIVE_CANDIDATE = None
            MODULE.BASE._ACTIVE_ATTEMPT = attempt
            MODULE.BASE._ACTIVE_CANDIDATE = "b" * 40
            try:
                MODULE.finalize_failure_triplet(RuntimeError("window"))
            finally:
                MODULE.BASE._ACTIVE_ATTEMPT = None
                MODULE.BASE._ACTIVE_CANDIDATE = None
            manifest = json.loads((attempt / "manifest.json").read_text())
            self.assertEqual(manifest["candidate_hash"], "b" * 40)
            self.assertEqual(manifest["verdict"], "FAIL")

    def test_terminal_pass_replays_frozen_scorer_and_binds_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-terminal-valid-") as temporary:
            attempt = Path(temporary)
            manifest, rows, summary = self.make_normal_terminal(attempt)
            self.assertTrue(MODULE.terminal_manifest_valid(attempt, "c" * 40))

            mutations = []
            mutations.append((dict(manifest), rows[:1], summary))
            mutations.append((dict(manifest), list(reversed(rows)), summary))
            forged = dict(summary)
            forged["observed"] = dict(forged["observed"], warm_append_seconds_saved=999.0)
            mutations.append((dict(manifest), rows, forged))
            for field, value in (
                ("binary_sha256", "e" * 64),
                ("engine_source_commit", "e" * 40),
                ("configuration_sha256", "e" * 64),
            ):
                changed = dict(manifest)
                changed[field] = value
                mutations.append((changed, rows, summary))
            for changed_manifest, changed_rows, changed_summary in mutations:
                self.rewrite_terminal(attempt, changed_manifest, changed_rows, changed_summary)
                self.assertFalse(MODULE.terminal_manifest_valid(attempt, "c" * 40))

    def test_failure_schema_can_never_claim_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-terminal-failure-pass-") as temporary:
            attempt = Path(temporary)
            raw = b""
            summary = b'{"verdict":"PASS"}\n'
            (attempt / "raw.jsonl").write_bytes(raw)
            (attempt / "summary.json").write_bytes(summary)
            manifest = {
                "schema": "glm52-w7-evict-store-probe-failure-v1",
                "candidate_hash": "d" * 40,
                "failure": "injected",
                "runner_sha256": hashlib.sha256(MODULE.Path(MODULE.__file__).read_bytes()).hexdigest(),
                "scorer_sha256": MODULE.SCORER_SHA256,
                "binary_sha256": MODULE.BINARY_SHA256,
                "model_sha256": MODULE.MODEL_SHA256,
                "completed_rows": 0,
                "artifacts": {
                    "raw.jsonl": hashlib.sha256(raw).hexdigest(),
                    "summary.json": hashlib.sha256(summary).hexdigest(),
                },
                "binding_failures": {},
                "verdict": "PASS",
            }
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            self.assertFalse(MODULE.terminal_manifest_valid(attempt, "d" * 40))

    def test_normal_publication_preserves_pass_and_fail_verdicts(self) -> None:
        for expected_verdict in ("PASS", "FAIL"):
            with tempfile.TemporaryDirectory(prefix="w7-publish-") as temporary:
                attempt = Path(temporary)
                manifest, rows, _ = self.make_normal_terminal(attempt)
                if expected_verdict == "FAIL":
                    on_row = next(row for row in rows if row["arm"] == "on")
                    on_row["token_timestamps_ns"] = [
                        on_row["token_timestamps_ns"][0] + index * 200_000_000
                        for index in range(128)
                    ]
                scorer_rows = []
                for row in rows:
                    scorer_row = dict(row)
                    scorer_row.pop("executed_environment_sha256")
                    scorer_rows.append(scorer_row)
                scored = MODULE.load_scorer(MODULE.SCORER.read_bytes()).score_probe_rows(
                    scorer_rows, manifest["arm_order"]
                )
                self.assertEqual(scored["verdict"], expected_verdict)
                raw = b"".join(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                    for row in rows
                )
                summary = (json.dumps(scored, sort_keys=True, separators=(",", ":")) + "\n").encode()
                for name in ("manifest.json", "raw.jsonl", "summary.json"):
                    (attempt / name).unlink()
                manifest["candidate_hash"] = "f" * 40
                manifest["artifacts"] = {}
                observed = MODULE.publish_terminal_triplet(attempt, manifest, raw, summary, {})
                self.assertEqual(observed, expected_verdict)
                self.assertTrue(MODULE.terminal_manifest_valid(attempt, "f" * 40))
                published = json.loads((attempt / "manifest.json").read_text())
                self.assertEqual(published["verdict"], expected_verdict)

    def test_failure_publication_records_authoritative_replacement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="w7-binding-finalize-") as temporary:
            attempt = Path(temporary)
            arm = attempt / "p0-off"
            arm.mkdir()
            server = arm / "server.log"
            server.write_text("receipt-bound\n")
            expected = hashlib.sha256(server.read_bytes()).hexdigest()
            server.write_text("replacement\n")
            MODULE._ACTIVE_ATTEMPT = attempt
            MODULE._ACTIVE_CANDIDATE = "d" * 40
            MODULE._ACTIVE_ROWS = [{"arm": "off"}]
            MODULE._ACTIVE_BINDINGS = {str(server): expected}
            try:
                MODULE.finalize_failure_triplet(RuntimeError("interrupted"))
            finally:
                MODULE._ACTIVE_ATTEMPT = None
                MODULE._ACTIVE_CANDIDATE = None
                MODULE._ACTIVE_ROWS = []
                MODULE._ACTIVE_BINDINGS = {}
            manifest = json.loads((attempt / "manifest.json").read_text())
            self.assertEqual(manifest["verdict"], "FAIL")
            self.assertIn("p0-off/server.log", manifest["binding_failures"])

    def test_parse_rejects_server_replacement_after_base_validation(self) -> None:
        temporary, out, base_row, receipt = self.make_parse_arm()
        self.addCleanup(temporary.cleanup)
        original = (out / "server.log").read_text()
        forged = original.replace("kv cache stored tokens=5055 trimmed=0 reason=evict key=token-text size=918.82 MiB save=700.0 ms\n", "")
        forged = forged.replace(
            "ds4-server: listening on http://127.0.0.1:8097\n",
            "ds4-server: listening on http://127.0.0.1:8097\n"
            + MODULE.ACTIVATION + "\n" + MODULE.SKIPPED + "\n",
        )
        def replace_after_validation(*_args, **_kwargs):
            (out / "server.log").write_text(forged)
            return base_row
        with mock.patch.object(MODULE.BASE, "parse_arm", side_effect=replace_after_validation):
            with self.assertRaises(Exception):
                MODULE.parse_arm("on", 0, out, 0, receipt, "4" * 64, "3" * 64)

    def test_parse_and_record_blocks_termination_until_row_and_bindings_publish(self) -> None:
        rows: list[dict[str, object]] = []
        bindings: dict[str, str] = {}
        def observe_mask(*_args, **_kwargs):
            current = MODULE.signal.pthread_sigmask(MODULE.signal.SIG_BLOCK, set())
            self.assertIn(MODULE.signal.SIGTERM, current)
            return {"arm": "off"}, {"/attempt/server.log": "a" * 64}
        with mock.patch.object(MODULE, "parse_arm", side_effect=observe_mask):
            MODULE.parse_and_record_arm(
                rows, bindings, "off", 0, Path("/attempt"), 0, "receipt",
                "b" * 64, "c" * 64, "d" * 64,
            )
        self.assertEqual(rows, [{"arm": "off", "executed_environment_sha256": "d" * 64}])
        self.assertEqual(bindings, {"/attempt/server.log": "a" * 64})

    def test_parse_accepts_only_exact_off_and_on_event_sets(self) -> None:
        temporary, out, base_row, off_receipt = self.make_parse_arm()
        self.addCleanup(temporary.cleanup)
        with mock.patch.object(MODULE.BASE, "parse_arm", return_value=copy.deepcopy(base_row)):
            off, _ = MODULE.parse_arm("off", 0, out, 0, off_receipt, "4" * 64, "3" * 64)
        self.assertEqual(off["evict_store_count"], 1)
        self.assertEqual(off["checkpoint_id"], "token-text:9e5ba8aa0b75e6c618f68d9834ef541c44cd4b42")
        on_receipt = self.convert_server_to_on(out)
        with mock.patch.object(MODULE.BASE, "parse_arm", return_value=copy.deepcopy(base_row)):
            on, _ = MODULE.parse_arm("on", 0, out, 0, on_receipt, "4" * 64, "3" * 64)
        self.assertEqual(on["evict_store_count"], 0)
        self.assertEqual(on["skip_marker_count"], 1)

    def test_parse_rejects_extra_or_substituted_event_payloads(self) -> None:
        temporary, out, base_row, _receipt = self.make_parse_arm()
        self.addCleanup(temporary.cleanup)
        with (out / "server.log").open("a") as stream:
            stream.write("ds4-server: kv cache stored tokens=4096 trimmed=0 reason=evict key=token-text size=1 MiB save=1 ms\n")
            stream.write("ds4-server: kv cache stored malformed reason=evict\n")
        receipt = self.bind_server(out)
        with mock.patch.object(MODULE.BASE, "parse_arm", return_value=base_row):
            with self.assertRaises(Exception):
                MODULE.parse_arm("off", 0, out, 0, receipt, "4" * 64, "3" * 64)

        temporary_on, out_on, base_on, _ = self.make_parse_arm()
        self.addCleanup(temporary_on.cleanup)
        self.convert_server_to_on(out_on)
        with (out_on / "server.log").open("a") as stream:
            stream.write("ds4-server: diagnostic skipped preload evict store live=4096 prompt=4100 common=4000\n")
            stream.write("ds4-server: diagnostic skipped preload evict store\n")
        on_receipt = self.bind_server(out_on)
        with mock.patch.object(MODULE.BASE, "parse_arm", return_value=base_on):
            with self.assertRaises(Exception):
                MODULE.parse_arm("on", 0, out_on, 0, on_receipt, "4" * 64, "3" * 64)

    def test_parse_rejects_arbitrary_prefix_on_each_expected_event(self) -> None:
        for target in ("store", "hit"):
            temporary, out, base_row, _ = self.make_parse_arm()
            self.addCleanup(temporary.cleanup)
            path = out / "server.log"
            source = path.read_text()
            needle = (
                "ds4-server: kv cache stored tokens=5055" if target == "store"
                else "ds4-server: kv cache hit text tokens=5044"
            )
            path.write_text(source.replace(needle, "MALFORMED_PREFIX " + needle))
            receipt = self.bind_server(out)
            with mock.patch.object(MODULE.BASE, "parse_arm", return_value=copy.deepcopy(base_row)):
                with self.assertRaises(Exception):
                    MODULE.parse_arm("off", 0, out, 0, receipt, "4" * 64, "3" * 64)
        for target in (MODULE.ACTIVATION, MODULE.SKIPPED):
            temporary, out, base_row, _ = self.make_parse_arm()
            self.addCleanup(temporary.cleanup)
            self.convert_server_to_on(out)
            path = out / "server.log"
            path.write_text(path.read_text().replace(target, "MALFORMED_PREFIX " + target))
            receipt = self.bind_server(out)
            with mock.patch.object(MODULE.BASE, "parse_arm", return_value=copy.deepcopy(base_row)):
                with self.assertRaises(Exception):
                    MODULE.parse_arm("on", 0, out, 0, receipt, "4" * 64, "3" * 64)

    def test_parse_rejects_logit_replacement_after_base_validation(self) -> None:
        temporary, out, base_row, receipt = self.make_parse_arm()
        self.addCleanup(temporary.cleanup)
        target = out / "logits.sync2.start5044.prompt5055.suffix11"
        def replace_after_validation(*_args, **_kwargs):
            target.write_bytes(b"replacement")
            return base_row
        with mock.patch.object(MODULE.BASE, "parse_arm", side_effect=replace_after_validation):
            with self.assertRaises(Exception):
                MODULE.parse_arm("off", 0, out, 0, receipt, "4" * 64, "3" * 64)


if __name__ == "__main__":
    unittest.main()
