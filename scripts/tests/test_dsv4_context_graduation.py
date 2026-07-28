#!/usr/bin/env python3
"""Contracts for guarded DeepSeek context graduation."""

from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts" / "57_dsv4_context_probe.py"
WORKER = ROOT / "scripts" / "58_dsv4_context_worker.sh"
SCHEDULER = ROOT / "scripts" / "59_schedule_dsv4_context.sh"
USER_SCHEDULER = ROOT / "scripts" / "60_schedule_dsv4_context_user.sh"
SCORER = ROOT / "scripts" / "62_score_dsv4_context.py"
LAUNCHER = ROOT / "scripts" / "21_serve_llamacpp.sh"
ATTESTOR_INSTALLER = ROOT / "scripts" / "63_install_context_attestor.sh"
ATTESTOR_SUBMIT = ROOT / "scripts" / "64_context_submit.sh"


def load_probe():
    spec = importlib.util.spec_from_file_location("dsv4_context_probe", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load context probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_scorer():
    spec = importlib.util.spec_from_file_location("dsv4_context_scorer", SCORER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load context scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContextProbeTests(unittest.TestCase):
    def test_fixture_has_exact_token_count_and_three_position_bands(self):
        probe = load_probe()
        tokenizer = probe.load_tokenizer()
        fixture = probe.build_fixture(tokenizer, 130_000, "7" * 64)
        self.assertEqual(probe.token_count(tokenizer, fixture["text"]), 130_000)
        positions = [record["position"] for record in fixture["records"]]
        self.assertLessEqual(positions[0], 130_000 // 4)
        self.assertTrue(130_000 // 4 < positions[1] < 3 * 130_000 // 4)
        self.assertGreaterEqual(positions[2], 3 * 130_000 // 4)
        self.assertEqual(len({record["value"] for record in fixture["records"]}), 3)
        self.assertNotIn(fixture["absent_value"], fixture["text"])

    def test_retrieval_validation_fails_closed(self):
        probe = load_probe()
        records = [
            {"case_id": "needle-0", "value": "RECORD_ALPHA_aaa"},
            {"case_id": "needle-1", "value": "RECORD_BRAVO_bbb"},
            {"case_id": "needle-2", "value": "RECORD_CHARLIE_ccc"},
        ]
        valid = (
            "RECORD_ALPHA_aaa, RECORD_BRAVO_bbb, "
            "RECORD_CHARLIE_ccc, NO_EXTRA_RECORD"
        )
        self.assertTrue(probe.validate_retrieval(valid, records)["pass"])
        payload_only = "aaa, bbb, ccc, NO_EXTRA_RECORD"
        self.assertTrue(
            probe.validate_retrieval(payload_only, records)["pass"]
        )
        verbose = (
            "AUDIT RECORD 1: RECORD_ALPHA_aaa\n"
            "AUDIT RECORD 2: RECORD_BRAVO_bbb\n"
            "AUDIT RECORD 3: RECORD_CHARLIE_ccc\n\n"
            + valid
        )
        self.assertTrue(probe.validate_retrieval(verbose, records)["pass"])
        absent = "RECORD_DELTA_absent"
        self.assertTrue(
            probe.validate_retrieval(valid, records, absent_value=absent)["pass"]
        )
        self.assertFalse(
            probe.validate_retrieval(
                valid + ", " + absent,
                records,
                absent_value=absent,
            )["pass"]
        )
        self.assertFalse(
            probe.validate_retrieval(valid.replace("BRAVO_bbb", "BRAVO_bad"), records)[
                "pass"
            ]
        )
        self.assertFalse(
            probe.validate_retrieval(valid + ", RECORD_DELTA_fake", records)["pass"]
        )
        self.assertFalse(
            probe.validate_retrieval(valid.replace("NO_EXTRA_RECORD", ""), records)[
                "pass"
            ]
        )

    def test_engine_progress_is_required_and_cannot_be_faked_by_usage(self):
        probe = load_probe()
        log = "\n".join(
            (
                "I slot print_timing: id 0 | task 7 | prompt processing, "
                "n_tokens = 999488, progress = 1.00",
                "I slot print_timing: id 0 | task 7 | prompt processing, "
                "n_tokens = 1000000, progress = 1.00",
            )
        )
        evidence = probe.parse_engine_progress(log, task_id=7)
        self.assertEqual(evidence["evaluated_tokens"], 1_000_000)
        with self.assertRaisesRegex(RuntimeError, "engine progress"):
            probe.parse_engine_progress("", task_id=7)
        with self.assertRaisesRegex(RuntimeError, "target"):
            probe.require_token_count_agreement(
                requested_tokens=1_000_000,
                usage_tokens=1_000_000,
                engine_tokens=999_488,
            )

    def test_only_final_content_can_satisfy_retrieval_and_stop_is_required(self):
        probe = load_probe()
        records = [
            {"case_id": "needle-0", "value": "RECORD_ALPHA_aaa"},
            {"case_id": "needle-1", "value": "RECORD_BRAVO_bbb"},
            {"case_id": "needle-2", "value": "RECORD_CHARLIE_ccc"},
        ]
        valid = (
            "RECORD_ALPHA_aaa, RECORD_BRAVO_bbb, "
            "RECORD_CHARLIE_ccc, NO_EXTRA_RECORD"
        )
        self.assertFalse(
            probe.validate_completion(
                content="",
                reasoning_content=valid,
                finish_reason="stop",
                done=True,
                records=records,
            )["pass"]
        )
        self.assertFalse(
            probe.validate_completion(
                content=valid,
                reasoning_content="",
                finish_reason="length",
                done=True,
                records=records,
            )["pass"]
        )
        self.assertTrue(
            probe.validate_completion(
                content=valid,
                reasoning_content="",
                finish_reason="stop",
                done=True,
                records=records,
            )["pass"]
        )

    def test_context_probe_freezes_non_thinking_request_mode(self):
        probe = load_probe()
        payload = probe.completion_payload("prompt", "1" * 64)
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["temperature"], 0)

    def test_context_probe_allows_only_one_terminal_eos_without_sse_text(self):
        probe = load_probe()
        self.assertEqual(
            probe.completed_text_token_count(usage_tokens=142, event_count=141),
            141,
        )
        self.assertEqual(
            probe.completed_text_token_count(usage_tokens=141, event_count=141),
            141,
        )
        with self.assertRaisesRegex(RuntimeError, "timestamp"):
            probe.completed_text_token_count(usage_tokens=142, event_count=140)

    def test_scorer_regenerates_seeded_request_and_rejects_stage_mutation(self):
        probe = load_probe()
        scorer = load_scorer()
        tokenizer = probe.load_tokenizer()
        expected = probe.build_request_artifacts(
            tokenizer, target=1024, seed_sha256="2" * 64
        )
        stage = {
            "seed_sha256": "2" * 64,
            "target_tokens": 1024,
            "fixture_sha256": expected["fixture"]["fixture_sha256"],
            "records": expected["fixture"]["records"],
            "absent_value_sha256": expected["absent_value_sha256"],
            "request_sha256": expected["request_sha256"],
        }
        scorer.validate_stage_lineage(stage, expected)
        for field, replacement in (
            ("fixture_sha256", "f" * 64),
            ("records", []),
            ("absent_value_sha256", "e" * 64),
            ("request_sha256", "d" * 64),
        ):
            broken = json.loads(json.dumps(stage))
            broken[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "lineage"
            ):
                scorer.validate_stage_lineage(broken, expected)

    def test_protected_binary_hash_runs_under_service_identity(self):
        scorer = load_scorer()
        completed = mock.Mock(stdout=("a" * 64 + "  /protected/binary\n"))
        with mock.patch.object(scorer.subprocess, "run", return_value=completed) as run:
            self.assertEqual(
                scorer.hash_as_dsv4(Path("/protected/binary")), "a" * 64
            )
        self.assertEqual(
            run.call_args.args[0],
            [
                "/usr/bin/sudo",
                "-n",
                "-u",
                "dsv4",
                "/usr/bin/sha256sum",
                "--",
                "/protected/binary",
            ],
        )

    def test_freeze_verification_uses_git_tree_and_rejects_added_files(self):
        scorer = load_scorer()
        candidate = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        archive = subprocess.check_output(
            ["git", "-C", str(ROOT), "archive", candidate]
        )
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory)
            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
                stream.extractall(frozen, filter="data")
            tokenizer = ROOT / "vendor" / "official-encoding" / "tokenizer.json"
            frozen_tokenizer = (
                frozen / "vendor" / "official-encoding" / "tokenizer.json"
            )
            frozen_tokenizer.parent.mkdir(parents=True, exist_ok=True)
            frozen_tokenizer.write_bytes(tokenizer.read_bytes())

            artifacts = {
                str(path.relative_to(frozen)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in frozen.rglob("*")
                if path.is_file()
            }
            manifest_path = frozen / "freeze-manifest.json"
            manifest = {
                "schema_version": 1,
                "candidate_hash": candidate,
                "artifacts": artifacts,
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            scorer.verify_freeze(frozen, candidate)

            target = frozen / "scripts" / "57_dsv4_context_probe.py"
            target.write_text("# substituted\n", encoding="utf-8")
            manifest["artifacts"][
                "scripts/57_dsv4_context_probe.py"
            ] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Git candidate"):
                scorer.verify_freeze(frozen, candidate)

            with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
                stream.extract("scripts/57_dsv4_context_probe.py", frozen)
            manifest["artifacts"][
                "scripts/57_dsv4_context_probe.py"
            ] = hashlib.sha256(target.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (frozen / "scripts" / "hashlib.py").write_text(
                "# import shadow\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "unlisted"):
                scorer.verify_freeze(frozen, candidate)

    def test_failed_context_scoring_always_emits_complete_triplet(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            scorer.write_failure_triplet(
                out=out,
                candidate="a" * 40,
                seed="b" * 64,
                mode="graduated",
                error=RuntimeError("missing stage"),
            )
            for name in ("manifest.json", "raw.jsonl", "summary.json"):
                self.assertTrue((out / name).is_file(), name)
            manifest = json.loads(
                (out / "manifest.json").read_text(encoding="utf-8")
            )
            raw = json.loads(
                (out / "raw.jsonl").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (out / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["gate"], "dsv4_reference_context")
            self.assertEqual(raw["record_type"], "context_failure")
            self.assertEqual(summary["verdict"], "FAIL")
            self.assertFalse(manifest["qualification_authority"])

    def test_scheduler_seed_is_derived_from_three_post_freeze_drand_relays(self):
        scorer = load_scorer()
        candidate = "a" * 40
        signature = "12" * 96
        randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        round_number = 10
        beacon = {
            "round": round_number,
            "randomness": randomness,
            "signature": signature,
        }
        lineage = scorer.capture_public_lineage(
            candidate=candidate,
            now=datetime.fromtimestamp(
                1_595_431_050 + (round_number - 1) * 30 + 1,
                timezone.utc,
            ),
            relay_fetcher=lambda _host, requested: (
                beacon if requested == round_number else {}
            ),
            commit_time_fetcher=lambda _candidate: "2020-07-22T00:00:00+00:00",
        )
        expected_seed = hashlib.sha256(
            f"{candidate}:{randomness}:dsv4_reference_context".encode()
        ).hexdigest()
        self.assertEqual(
            lineage["randomness"]["seed_sha256"], expected_seed
        )
        self.assertEqual(lineage["randomness"]["round"], round_number)
        scheduler = USER_SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("[[ $SEED_SHA256 == auto ]]", scheduler)
        worker = WORKER.read_text(encoding="utf-8")
        self.assertNotIn("--capture-lineage", scheduler)
        self.assertIn("--capture-lineage", worker)

    def test_journal_witness_is_process_linked_and_tamper_evident(self):
        scorer = load_scorer()
        payload = {
            "event": "context-test-witness",
            "candidate_hash": "c" * 40,
            "nonce": "d" * 32,
        }
        receipt = scorer.emit_journal_witness(payload)
        scorer.verify_journal_witness(payload, receipt)
        tampered = dict(payload)
        tampered["candidate_hash"] = "e" * 40
        with self.assertRaisesRegex(RuntimeError, "journal witness"):
            scorer.verify_journal_witness(tampered, receipt)
        self.assertEqual(receipt["uid"], "1000")
        self.assertTrue(receipt["boot_id"])
        self.assertTrue(receipt["scope_id"])

    def test_security_subprocesses_ignore_ambient_path_and_python_hooks(self):
        scorer = load_scorer()
        completed = mock.Mock(stdout=("a" * 64 + "  /protected/binary\n"))
        with mock.patch.object(scorer.subprocess, "run", return_value=completed) as run:
            scorer.hash_as_dsv4(Path("/protected/binary"))
        self.assertEqual(run.call_args.args[0][0], "/usr/bin/sudo")
        self.assertIn("/usr/bin/sha256sum", run.call_args.args[0])
        self.assertEqual(
            run.call_args.kwargs["env"],
            {
                "HOME": "/nonexistent",
                "PATH": "/usr/bin:/bin",
                "LANG": "C.UTF-8",
            },
        )
        worker = WORKER.read_text(encoding="utf-8")
        scheduler = USER_SCHEDULER.read_text(encoding="utf-8")
        self.assertIn('PYTHONNOUSERSITE=1', worker)
        self.assertIn('PYTHONPATH=', worker)
        self.assertIn("/usr/bin/python3 -I", worker)
        self.assertNotIn(".venv-harness/bin/python", worker)
        self.assertNotIn(".venv-harness/bin/python", scheduler)

    def test_journal_artifact_witness_rejects_post_seal_rewrite(self):
        scorer = load_scorer()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stage.json").write_text(
                '{"pass":true}\n', encoding="utf-8"
            )
            (root / "engine.log").write_text(
                "completed batch\n", encoding="utf-8"
            )
            witness = scorer.create_artifact_witness(
                root=root,
                event="stage-complete",
                candidate="a" * 40,
                seed="b" * 64,
                artifacts=["stage.json", "engine.log"],
            )
            scorer.verify_artifact_witness(root=root, witness=witness)
            (root / "stage.json").write_text(
                '{"pass":false}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "artifact witness"):
                scorer.verify_artifact_witness(root=root, witness=witness)

    def test_pass_finalizer_recomputes_formula_and_seals_authority(self):
        scorer = load_scorer()
        candidate = "a" * 40
        seed = "b" * 64
        observation = {"record_type": "context_observation"}
        result = {
            "manifest": {
                "candidate_hash": candidate,
                "seed_sha256": seed,
                "freeze_witness": {
                    "receipt": {"invocation_id": "c" * 32}
                },
            },
            "summary": {"verdict": "PASS"},
        }
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "raw.jsonl").write_text(
                json.dumps(observation) + "\n", encoding="utf-8"
            )
            for name in ("manifest", "summary"):
                (out / f"{name}.json").write_text(
                    json.dumps(result[name]) + "\n", encoding="utf-8"
                )
            fake = dict(result["summary"])
            fake["fabricated"] = True
            (out / "summary.json").write_text(
                json.dumps(fake) + "\n", encoding="utf-8"
            )
            with (
                mock.patch.object(
                    scorer,
                    "require_qualification_invocation",
                    return_value="c" * 32,
                ),
                mock.patch.object(
                    scorer,
                    "build_observation",
                    return_value=(observation, result),
                ) as rebuild,
            ):
                with self.assertRaisesRegex(RuntimeError, "persisted summary"):
                    scorer.finalize_attempt(
                        out=out,
                        candidate=candidate,
                        seed=seed,
                        mode="graduated",
                        lifecycle_exit_status=0,
                    )
            rebuild.assert_called_once()
            self.assertFalse((out / "journal-seal.json").exists())
            self.assertFalse((out / "qualification.json").exists())

    def test_qualification_witness_requires_exact_worker_unit(self):
        scorer = load_scorer()
        with mock.patch.dict(scorer.os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "INVOCATION_ID"):
                scorer.require_qualification_invocation()
        with mock.patch.dict(
            scorer.os.environ, {"INVOCATION_ID": "a" * 32}, clear=True
        ):
            with (
                mock.patch.object(scorer.os, "geteuid", return_value=0),
                mock.patch.object(
                    scorer.Path,
                    "read_text",
                    return_value=(
                        "0::/user.slice/user-1000.slice/session-1.scope\n"
                    ),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "worker cgroup"):
                    scorer.require_qualification_invocation()


class ContextWorkerContractTests(unittest.TestCase):
    def test_root_attestor_replaces_unrestricted_dsv4_delegation(self):
        installer = ATTESTOR_INSTALLER.read_text(encoding="utf-8")
        submit = ATTESTOR_SUBMIT.read_text(encoding="utf-8")
        self.assertIn("/etc/sudoers.d/dsv4-delegate", installer)
        self.assertIn("/usr/local/sbin/dsv4-context-submit", installer)
        self.assertIn("git", installer)
        self.assertIn("CANDIDATE_HASH", installer)
        self.assertIn("SUBMIT_SHA256", installer)
        self.assertNotIn(
            '"$REPO/scripts/64_context_submit.sh" "$SUBMIT"', installer
        )
        self.assertIn("/usr/sbin/visudo -cf", installer)
        self.assertIn("NOPASSWD: /usr/local/sbin/dsv4-context-submit", installer)
        self.assertNotIn("(dsv4) NOPASSWD: ALL", installer)
        self.assertIn("(( EUID == 0 ))", submit)
        self.assertIn("readonly STATE_ROOT=/var/lib/dsv4-context", submit)
        self.assertIn("CANDIDATE_ROOT=$STATE_ROOT/candidates", submit)
        self.assertIn("ATTEMPT_ROOT=$STATE_ROOT/attempts", submit)
        self.assertIn("MODEL_ROOT=$STATE_ROOT/models/deepseek-v4-flash", submit)
        self.assertIn("/usr/bin/ln", submit)
        self.assertIn("/usr/bin/chown root:root", submit)
        self.assertIn('"$FROZEN/scripts/21_serve_llamacpp.sh" status', submit)
        self.assertNotIn(
            '"$REPO/scripts/21_serve_llamacpp.sh" status', submit
        )
        self.assertIn("/usr/bin/git", submit)
        self.assertIn("RuntimeMaxSec=43200", submit)
        self.assertIn("MemorySwapMax=0", submit)
        self.assertIn("systemd-run", submit)
        self.assertNotIn("systemd-run --user", submit)

    def test_preload_admission_allows_measured_headless_baseline(self):
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn("--required-gib 115.0", worker)
        self.assertNotIn("--required-gib 115.25", worker)

    def test_memory_telemetry_covers_engine_startup(self):
        worker = WORKER.read_text(encoding="utf-8")
        telemetry = worker.index('>>"$OUT/memory.jsonl" &')
        startup = worker.index('dsv4_launcher "$cap" 3 start')
        self.assertLess(telemetry, startup)

    def test_user_runner_needs_no_root_and_serializes_with_guard(self):
        worker = WORKER.read_text(encoding="utf-8")
        scheduler = USER_SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("sudo -n -u dsv4", worker)
        self.assertIn("/home/dsv4/.dsv4-start-hold", worker)
        self.assertIn("systemctl is-active --quiet dsv4-guard.service", worker)
        self.assertIn("DSV4_START_HOLD_FILE", worker)
        self.assertIn("DSV4_ALLOW_RETRY_AFTER_FAILED_START", worker)
        self.assertIn("--required-gib 110", worker)
        self.assertIn("systemd-run --user", scheduler)
        self.assertIn("dsv4-context-graduation.service", scheduler)
        self.assertIn("RuntimeMaxSec=43200", scheduler)
        self.assertIn("MemorySwapMax=0", scheduler)
        self.assertIn("TimeoutStopSec=600", scheduler)
        self.assertIn("frozen-candidate", scheduler)
        self.assertIn("sudo -n -u dsv4 chmod -R a-w", scheduler)
        self.assertIn("u:bmarti44:rX", scheduler)
        self.assertIn("--working-directory=", scheduler)
        self.assertNotIn("--directory=", scheduler)
        self.assertIn("repository is not clean", scheduler)
        self.assertIn("candidate hash changed", scheduler)
        self.assertNotIn("must run as root", scheduler)

    def test_user_probe_writes_evidence_as_evidence_owner(self):
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn("run_context_probe()", worker)
        self.assertIn("HOME=/home/bmarti44 USER=bmarti44", worker)
        self.assertIn('run_context_probe "$cap" "$target"', worker)

    def test_scheduler_preflight_failures_emit_triplet(self):
        scheduler = USER_SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("trap record_scheduler_failure EXIT", scheduler)
        self.assertIn("--record-preflight-failure", scheduler)
        self.assertIn("SCHEDULED=true", scheduler)

    def test_measured_headless_admission_is_default_off_and_bounded(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("DSV4_MEASURED_HEADLESS_OVERHEAD_GIB", source)
        self.assertIn("${DSV4_MEASURED_HEADLESS_OVERHEAD_GIB:-0}", source)
        self.assertIn("must be 0 or 3", source)
        self.assertIn("systemctl is-active --quiet display-manager.service", source)
        self.assertIn("overhead_gib=3", source)

    def test_14_gib_floor_is_restricted_to_headless_1m_qualification(self):
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB", source)
        self.assertIn("${DSV4_CONTEXT_QUALIFICATION_FLOOR_GIB:-0}", source)
        self.assertIn("must be 0 or 14", source)
        self.assertIn("CTX != 1048576", source)
        self.assertIn("measured_headless_overhead_gib != 3", source)
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn("floor=14", worker)
        self.assertIn("qualification_floor=14", worker)
        self.assertIn("DSV4_WATCHDOG_FLOOR_GIB=\"$floor\"", worker)
        self.assertIn("DSV4_MEM_FLOOR_GIB=\"$floor\"", worker)

    def test_1m_uses_smaller_prefill_buffers_and_external_restore_unit(self):
        worker = WORKER.read_text(encoding="utf-8")
        self.assertIn("batch=512", worker)
        self.assertIn("ubatch=256", worker)
        self.assertIn('DSV4_UBATCH="$ubatch"', worker)
        self.assertIn('DSV4_BATCH="$batch"', worker)
        self.assertIn("systemctl restart dsv4-engine-restore.service", worker)
        self.assertNotIn(
            "systemctl --no-block restart dsv4-engine-restore.service",
            worker,
        )
        restore = worker.index("systemctl restart dsv4-engine-restore.service")
        timer = worker.index("systemctl start dsv4-guard.timer", restore)
        self.assertLess(restore, timer)

    def test_worker_graduates_exact_rungs_and_restores_headless_safe_profile(self):
        source = WORKER.read_text(encoding="utf-8")
        self.assertIn("131072 262144 524288 1048576", source)
        self.assertIn("130000 260000 520000 1000000", source)
        self.assertIn("DSV4_WATCHDOG_FLOOR_GIB=18", source)
        self.assertIn("DSV4_MEM_FLOOR_GIB=18", source)
        self.assertIn("CTX=8192", source)
        self.assertIn("restore_safe_profile", source)
        self.assertIn("trap cleanup EXIT", source)
        self.assertIn("sleep 0.25", source)
        self.assertIn("journalctl -k --after-cursor", source)
        self.assertIn("NV_ERR_NO_MEMORY", source)
        self.assertIn("NVRM.*Xid", source)
        self.assertIn("oom-kill", source)
        self.assertIn("DSV4_SPEC_TYPE=none", source)
        self.assertNotIn("glm_safe_run.sh", source)
        self.assertNotIn("gguf-glm", source)
        self.assertNotIn("systemctl start display-manager.service", source)
        self.assertIn("swap_current_bytes", source)
        self.assertIn("w11.context.v1", source)
        self.assertNotIn('"verdict": "PASS" if (', source)

    def test_scheduler_is_detached_candidate_bound_and_serialized(self):
        source = SCHEDULER.read_text(encoding="utf-8")
        self.assertIn("must run as root", source)
        self.assertIn("repository is not clean", source)
        self.assertIn("candidate hash changed", source)
        self.assertIn("systemd-run", source)
        self.assertIn("--no-block", source)
        self.assertIn("dsv4-context-graduation.service", source)
        self.assertIn("RuntimeMaxSec=14400", source)
        self.assertIn("/run/dsv4/inference.lock", source)
        self.assertIn("glm52.process.json", source)
        self.assertNotIn("OnFailure=display-manager.service", source)
        stop_timer = source.index("systemctl stop dsv4-guard.timer")
        stop_guard = source.index("systemctl stop dsv4-guard.service", stop_timer)
        schedule = source.index("systemd-run", stop_guard)
        self.assertLess(stop_timer, stop_guard)
        self.assertLess(stop_guard, schedule)


if __name__ == "__main__":
    unittest.main()
