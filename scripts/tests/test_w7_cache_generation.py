#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/89_score_w7_cache_generation.py"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
SMOKE = ROOT / "results/glm52-gates/harness/w7_cache_generation_smoke_v1.sh"
SPEC = importlib.util.spec_from_file_location("w7_cache_generation_scorer", SCORER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


GOOD = """\
ds4: CUDA backend initialized
ds4: CUDA stable model remap enabled generation=1
0807 15:10:02 ds4-server: listening on http://127.0.0.1:8097
0807 15:10:03 ds4-server: completion ctx=5044..5066:22 prompt start
ds4: CUDA persistent expert cache enabled: 4110 slots x 9.28 MiB
ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1
0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt done 3.500s
0807 15:10:07 ds4-server: shutdown requested, draining requests
"""
HTTP = "200\n"
RESPONSE = (
    '{"choices":[{"finish_reason":"length","text":""}],'
    '"usage":{"prompt_tokens":5066,"completion_tokens":0,"total_tokens":5066,'
    '"prompt_tokens_details":{"cached_tokens":5044,"cache_write_tokens":22}}}\n'
)
RC = "0\n"
CONTAINMENT = (
    "SAFE_RUN_DONE rc=0 killed=no dir=/home/bmarti44/.local/state/glm52-crashlog/w7-test "
    f"main_sha256={'1' * 64} samples_sha256={'2' * 64} kernel_sha256={'3' * 64}\n"
)
BINARY_SHA256 = "eec10ca8aae5ef685e5420b02a56a1b76afaac9416acd58efb4230b15678a4d2"
ENV_SHA256 = "ea8cc542bf2138646cb5bb3d38c9f7e7d88eef3e5a8fe7faf13074463f5a5e64"
SAFETY = (
    "SAFE_RUN start tag=w7-test vlimit_kb=419430400 kill_floor_gib=24 "
    "min_start_gib=110 timeout_s=2400\n"
    "cgroup_verified path=/x memory_high=83751862272 memory_max=85899345920 "
    "memory_swap_max=0 memory_oom_group=1\n"
    f"executed_environment_allowlist=DS4_CUDA_STABLE_MODEL_REMAP executed_environment_sha256={ENV_SHA256}\n"
    f"executed_candidate_verified executed_binary_sha256={BINARY_SHA256}\n"
    "executed candidate was verified alive at least once; no identity contradiction observed by the periodic sampler\n"
    "SAFE_RUN end rc=0 killed=no\n"
)
MODEL_SHA256 = "a49de64c5020432bdae23de36a423a9660a5621bc0db8d12b66bd8814b07fea0"
MODEL_IDENTITY = (
    '{"bytes":211075856448,"device":1,"inode":2,'
    f'"sha256":"{MODEL_SHA256}","executed_path":"/proc/123/fd/10"}}'
)


def score(text: str = GOOD, *, http: str = HTTP, response: str = RESPONSE,
          rc: str = RC, containment: str = CONTAINMENT) -> dict[str, object]:
    return MODULE.score_text(
        text,
        http_status=http,
        response_text=response,
        containment_rc=rc,
        containment_stdout=containment,
        mode="on",
        child_exit_text='{"shutdown_requested":true,"forced_kill":false,"exit_status":0}',
        safety_main_text=SAFETY,
        expected_binary_sha256=BINARY_SHA256,
        expected_environment_sha256=ENV_SHA256,
        model_identity_text=MODEL_IDENTITY,
        expected_model_sha256=MODEL_SHA256,
        expected_model_bytes=211075856448,
    )


class W7CacheGenerationGateTest(unittest.TestCase):
    def test_driver_requires_safe_wrapper_ancestor_lineage(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        driver_entry = source.split('if [[ ${1:-} == --driver ]]', 1)[1].split("fi", 1)[0]
        self.assertIn("verify_driver_safe_lineage", driver_entry)
        self.assertLess(
            driver_entry.index("verify_driver_safe_lineage"),
            driver_entry.index("run_driver"),
        )
        self.assertIn("GLM_SAFE_W7_DRIVER_LINEAGE", source)

    def test_candidate_resolution_disables_git_replacements(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        self.assertIn("--no-replace-objects", source)
        for function in ("verify_reviewed_sources", "verify_sealed_candidate_scripts"):
            body = source.split(f"{function}() {{", 1)[1].split("\n}", 1)[0]
            self.assertIn("--no-replace-objects", body)

    def test_final_head_change_is_fail_closed_and_observed(self) -> None:
        scorer = SCORER.read_text(encoding="utf-8")
        harness = SMOKE.read_text(encoding="utf-8")
        self.assertIn("final_head_matches_candidate", scorer)
        self.assertIn("observed_final_head", harness)
        publisher = harness.split("publish_failure_triplet() {", 1)[1].split("finalize_outer() {", 1)[0]
        self.assertNotIn('"execution_head":candidate', publisher)

    def test_exact_byte_unsealed_harness_is_rejected(self) -> None:
        descriptor = os.memfd_create("w7-unsealed-harness", os.MFD_ALLOW_SEALING)
        try:
            payload = SMOKE.read_bytes()
            os.write(descriptor, payload)
            os.lseek(descriptor, 0, os.SEEK_SET)
            path = f"/proc/{os.getpid()}/fd/{descriptor}"
            env = dict(os.environ)
            env["DS4_W7_PINNED_HARNESS_SHA256"] = hashlib.sha256(payload).hexdigest()
            completed = subprocess.run(
                ["/usr/bin/bash", path, "--self-test"],
                pass_fds=(descriptor,),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
        finally:
            os.close(descriptor)

    def test_alternate_entry_binds_seals_and_candidate_objects(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        self.assertIn("verify_sealed_candidate_scripts", source)
        self.assertIn("F_GET_SEALS", source)
        self.assertIn('"$candidate:$tracked"', source)
        self.assertIn('/usr/bin/git -C "$ROOT" rev-parse HEAD', source)
        self.assertIn('/usr/bin/git -C "$ROOT" show', source)
        self.assertIn("--sealed-outer", source)
        self.assertIn("--driver", source)

    def test_failure_finalizer_is_independent_of_seal_holder(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        finalizer = source.split("finalize_outer() {", 1)[1].split("if [[ ${1:-} == --self-test", 1)[0]
        self.assertNotIn("scorer_fd_path", finalizer)
        self.assertNotIn("SourceFileLoader", finalizer)
        self.assertIn("publish_failure_triplet", finalizer)
        self.assertIn('containment_stdout="$containment_stdout"', source)
        self.assertIn('containment_rc="$containment_rc"', source)

    def test_holder_birth_identity_is_bound_before_cleanup(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        self.assertIn("seal_holder_start_ticks", source)
        self.assertIn("seal_holder_parent_pid", source)
        self.assertIn("verify_seal_holder_identity", source)
        cleanup = source.split("stop_seal_holder() {", 1)[1].split("verify_driver_containment() {", 1)[0]
        self.assertIn("verify_seal_holder_identity", cleanup)

    def test_sealed_runtime_snapshots_reject_in_place_mutation(self) -> None:
        completed = subprocess.run(
            [str(SMOKE), "--sealed-self-test"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("W7_SEALED_SNAPSHOTS_SELFTEST_OK", completed.stdout)

    def test_requests_are_submitted_from_sealed_snapshots(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        self.assertIn('live_fd_path="/proc/$seal_holder_pid/fd/', source)
        self.assertIn('primary_fd_path="/proc/$seal_holder_pid/fd/', source)
        self.assertIn('-d @"$live_fd_path"', source)
        self.assertIn('-d @"$primary_fd_path"', source)
        driver = source.split("run_driver() {", 1)[1].split("publish_outer_evidence() {", 1)[0]
        self.assertNotIn('-d @"$LIVE"', driver)
        self.assertNotIn('-d @"$PRIMARY"', driver)

    def test_completion_record_is_passed_from_private_capture(self) -> None:
        parameters = inspect.signature(MODULE.score_and_publish_bound_attempt).parameters
        self.assertIn("containment_stdout", parameters)
        self.assertIn("containment_rc", parameters)
        scorer = inspect.getsource(MODULE.score_and_publish_bound_attempt)
        self.assertNotIn('attempt / "containment.stdout"', scorer)
        self.assertNotIn('attempt / "containment.rc"', scorer)
        harness = SMOKE.read_text(encoding="utf-8")
        self.assertIn("containment_stdout=$(", harness)
        self.assertIn('"$containment_rc" "$containment_stdout"', harness)
        self.assertIn("containment_stdout=containment_stdout", harness)
        self.assertIn("containment_rc=containment_rc", harness)

    def test_private_completion_capture_ignores_replaced_persisted_file(self) -> None:
        crash_parent = Path("/home/bmarti44/.local/state/glm52-crashlog")
        crash_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="w7-private-") as directory, \
             tempfile.TemporaryDirectory(prefix="w7-private-", dir=crash_parent) as crash_directory:
            attempt = Path(directory) / "attempt"
            out = attempt / "on"
            crash = Path(crash_directory)
            out.mkdir(parents=True)
            runtime = {
                "server.log": GOOD,
                "live-response.json": RESPONSE,
                "live-http-status": HTTP,
                "primary-response.json": RESPONSE,
                "primary-http-status": HTTP,
                "child-exit.json": '{"shutdown_requested":true,"forced_kill":false,"exit_status":0}',
                "model.identity.json": MODEL_IDENTITY,
            }
            for name, payload in runtime.items():
                (out / name).write_text(payload)
            (attempt / "containment.stderr").write_text("")
            (attempt / "containment.stdout").write_text("fabricated persisted completion\n")
            (crash / "samples.log").write_text("samples\n")
            (crash / "kernel.log").write_text("kernel\n")
            final_lines = []
            for name in runtime:
                path = out / name
                metadata = path.stat()
                final_lines.append(
                    f"2026-08-07 final_artifact_verified path={path} "
                    f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()} "
                    f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}"
                )
            samples_hash = hashlib.sha256((crash / "samples.log").read_bytes()).hexdigest()
            kernel_hash = hashlib.sha256((crash / "kernel.log").read_bytes()).hexdigest()
            main = SAFETY + "\n".join(final_lines) + "\n" + (
                f"2026-08-07 safety_artifact_verified name=samples.log sha256={samples_hash} size=8\n"
                f"2026-08-07 safety_artifact_verified name=kernel.log sha256={kernel_hash} size=7\n"
            )
            (crash / "main.log").write_text(main)
            main_hash = hashlib.sha256((crash / "main.log").read_bytes()).hexdigest()
            private_completion = (
                f"SAFE_RUN_DONE rc=0 killed=no dir={crash} main_sha256={main_hash} "
                f"samples_sha256={samples_hash} kernel_sha256={kernel_hash}"
            )
            identities = {
                "candidate_hash": "a" * 40,
                "execution_head": "a" * 40,
                "binary_sha256": BINARY_SHA256,
                "model_sha256": MODEL_SHA256,
                "model_bytes": 211075856448,
                "live_request_sha256": "1" * 64,
                "primary_request_sha256": "2" * 64,
                "executed_environment_sha256": ENV_SHA256,
                "scorer_sha256": "3" * 64,
                "harness_sha256": "4" * 64,
                "cgroup_sha256": "5" * 64,
                "safe_run_sha256": "6" * 64,
                "containment": {},
            }
            result = MODULE.score_and_publish_bound_attempt(
                attempt=attempt,
                out=out,
                crash_dir=crash,
                evidence_dir=out / "evidence",
                identities=identities,
                containment_stdout=private_completion,
                containment_rc=0,
            )
            self.assertEqual(result["verdict"], "PASS")
            manifest = json.loads((out / "evidence/manifest.json").read_text())
            self.assertEqual(
                manifest["artifacts"]["containment.stdout"],
                hashlib.sha256(private_completion.encode()).hexdigest(),
            )
            self.assertNotEqual(
                manifest["artifacts"]["containment.stdout"],
                hashlib.sha256((attempt / "containment.stdout").read_bytes()).hexdigest(),
            )

    def test_rejects_copied_launcher_before_self_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copied-smoke.sh"
            shutil.copy2(SMOKE, copied)
            completed = subprocess.run(
                [str(copied), "--self-test"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_spoofed_direct_driver_fails_before_runtime_scratch(self) -> None:
        base = Path("/home/bmarti44/.local/state/glm52-w7-cache-generation")
        out = base / f"attempt-{uuid.uuid4().hex}" / "on"
        out.mkdir(parents=True, mode=0o700)
        env = dict(os.environ)
        env.update(GLM_SAFE_REQUIRE_CGROUP="1", DS4_CUDA_STABLE_MODEL_REMAP="1")
        try:
            subprocess.run(
                ["timeout", "--kill-after=1", "0.2", str(SMOKE), "--driver", "on", str(out)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            self.assertFalse((out / "kv").exists())
        finally:
            shutil.rmtree(out.parent, ignore_errors=True)

    def test_atomic_triplet_binds_raw_and_survives_no_runtime_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "evidence"
            summary = {"checks": {"runtime_completed": False}, "verdict": "FAIL"}
            raw = [{"source": "outer", "containment_rc": 8}]
            manifest = {"schema": "test", "artifacts": {}}
            MODULE.publish_triplet_atomic(destination, manifest, raw, summary)
            raw_bytes = (destination / "raw.jsonl").read_bytes()
            published = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(
                published["artifacts"]["raw.jsonl"],
                hashlib.sha256(raw_bytes).hexdigest(),
            )
            self.assertEqual(json.loads((destination / "summary.json").read_text())["verdict"], "FAIL")
            with self.assertRaises(FileExistsError):
                MODULE.publish_triplet_atomic(destination, manifest, raw, summary)

    def test_bound_copy_rejects_symlink_and_wrong_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.write_bytes(b"authoritative\n")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            MODULE.copy_bound_artifact(source, root / "copy", digest)
            self.assertEqual((root / "copy").read_bytes(), source.read_bytes())
            with self.assertRaises(ValueError):
                MODULE.copy_bound_artifact(source, root / "wrong", "0" * 64)
            link = root / "link"
            link.symlink_to(source)
            with self.assertRaises(OSError):
                MODULE.copy_bound_artifact(link, root / "linked-copy", digest)

    def test_runtime_binding_rejects_post_wrapper_replacement(self) -> None:
        crash_parent = Path("/home/bmarti44/.local/state/glm52-crashlog")
        crash_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="w7-bind-", dir=crash_parent) as directory:
            root = Path(directory)
            attempt, out, crash = root / "attempt", root / "attempt/on", root
            out.mkdir(parents=True)
            names = (
                "server.log", "live-response.json", "live-http-status",
                "primary-response.json", "primary-http-status", "child-exit.json",
                "model.identity.json",
            )
            lines = []
            for index, name in enumerate(names):
                path = out / name
                path.write_bytes(f"artifact-{index}\n".encode())
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                metadata = path.stat()
                lines.append(
                    f"2026-08-07 final_artifact_verified path={path} sha256={digest} "
                    f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}"
                )
            (crash / "samples.log").write_text("samples\n")
            (crash / "kernel.log").write_text("kernel\n")
            sample_hash = hashlib.sha256((crash / "samples.log").read_bytes()).hexdigest()
            kernel_hash = hashlib.sha256((crash / "kernel.log").read_bytes()).hexdigest()
            lines.extend((
                f"2026-08-07 safety_artifact_verified name=samples.log sha256={sample_hash} size=8",
                f"2026-08-07 safety_artifact_verified name=kernel.log sha256={kernel_hash} size=7",
            ))
            (crash / "main.log").write_text("\n".join(lines) + "\n")
            main_hash = hashlib.sha256((crash / "main.log").read_bytes()).hexdigest()
            (attempt / "containment.stdout").write_text(
                f"SAFE_RUN_DONE rc=0 killed=no dir={crash} main_sha256={main_hash} "
                f"samples_sha256={sample_hash} kernel_sha256={kernel_hash}\n"
            )
            (attempt / "containment.stderr").write_text("")
            (attempt / "containment.rc").write_text("0\n")
            replacement = out / "server.log.replacement"
            replacement.write_bytes((out / "server.log").read_bytes())
            os.replace(replacement, out / "server.log")
            with self.assertRaises(ValueError):
                MODULE.bind_runtime_artifacts(attempt, out, crash)

    def test_containment_forwards_exact_stable_remap_flag(self) -> None:
        source = CGROUP.read_text(encoding="utf-8")
        self.assertIn("DS4_CUDA_STABLE_MODEL_REMAP", source)
        self.assertIn("GLM_SAFE_DONE_DIGESTS", source)

    def test_smoke_uses_reviewed_binary_and_hardened_production_path(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        self.assertIn('/usr/bin/bash "$cgroup_fd_path" --tag', source)
        self.assertIn('"$harness_fd_path" --driver', source)
        self.assertIn("verify_driver_containment", source)
        self.assertIn("GLM_SAFE_FINAL_ARTIFACTS", source)
        self.assertIn("trap finalize_outer EXIT", source)
        for setting in (
            "GLM_SAFE_MEMORY_HIGH_GIB=78",
            "GLM_SAFE_KILL_FLOOR_GIB=24",
            "GLM_SAFE_MIN_START_GIB=110",
            "GLM_SAFE_TIMEOUT_S=2400",
            "GLM_SAFE_RUN_AS_CURRENT_USER=1",
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE=1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256",
        ):
            self.assertIn(setting, source)
        self.assertIn("w7-stable-remap-bccf0b6/ds4-server", source)
        self.assertIn("readonly BINARY_SHA256=", source)
        self.assertIn("readonly MODEL_SHA256=", source)
        self.assertIn("DS4_CUDA_STABLE_MODEL_REMAP", source)
        self.assertIn("--ssd-streaming-cache-experts 40GB", source)
        self.assertIn("--kv-cache-boundary-align-tokens 4", source)
        self.assertIn("--kv-cache-boundary-trim-tokens 8", source)
        self.assertIn('trap cleanup_driver EXIT INT TERM HUP', source)
        self.assertNotIn("kill -KILL", source)
        self.assertIn("child-exit.json", source)
        evidence_source = source + SCORER.read_text(encoding="utf-8")
        for evidence in ("manifest.json", "raw.jsonl", "summary.json"):
            self.assertIn(evidence, evidence_source)
        self.assertIn('sync -f "$out"', source)

    def test_runtime_scripts_execute_through_pinned_descriptors(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        self.assertIn('harness_fd_path="/proc/$seal_holder_pid/fd/$harness_fd"', source)
        self.assertIn('cgroup_fd_path="/proc/$seal_holder_pid/fd/$cgroup_fd"', source)
        self.assertIn('safe_fd_path="/proc/$seal_holder_pid/fd/$safe_fd"', source)
        self.assertIn('scorer_fd_path="/proc/$seal_holder_pid/fd/$scorer_fd"', source)
        self.assertIn("F_ADD_SEALS", source)
        self.assertIn("F_SEAL_WRITE", source)
        self.assertIn('"$harness_fd_path" --driver', source)

    def test_scoring_and_publication_share_one_immutable_snapshot(self) -> None:
        harness = SMOKE.read_text(encoding="utf-8")
        scorer = SCORER.read_text(encoding="utf-8")
        self.assertIn("score_and_publish_bound_attempt", scorer)
        self.assertIn("score_and_publish_bound_attempt", harness)
        self.assertNotIn('$out/bound/server.log', harness)

    def test_snapshot_bytes_cannot_be_reopened_after_path_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "server.log"
            source.write_text(GOOD.replace("shutdown requested, draining requests\n", ""))
            expected = source.stat()
            payload, _ = MODULE._read_snapshot(
                source,
                expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                expected_identity=(expected.st_dev, expected.st_ino, expected.st_size),
            )
            replacement = source.with_suffix(".replacement")
            replacement.write_text(GOOD)
            os.replace(replacement, source)
            self.assertEqual(score(payload.decode())["verdict"], "FAIL")
            self.assertEqual(score(source.read_text())["verdict"], "PASS")

    def test_failure_finalizer_uses_full_frozen_manifest_schema(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        finalizer = source.split("publish_failure_triplet() {", 1)[1].split("if [[ ${1:-} == --self-test", 1)[0]
        for field in (
            "binary_sha256", "model_sha256", "scorer_sha256", "harness_sha256",
            "cgroup_sha256", "safe_run_sha256", "live_request_sha256",
            "primary_request_sha256", "executed_environment_sha256", "containment",
        ):
            self.assertIn(field, finalizer)

    def test_scorer_binds_on_activation_and_child_exit(self) -> None:
        source = SCORER.read_text(encoding="utf-8")
        self.assertIn("CUDA stable model remap enabled generation=", source)
        self.assertIn("child_exit", source)
        self.assertIn("shutdown_observed_once", source)

    def test_accepts_completed_resume_without_false_reload(self) -> None:
        result = score()
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["observed"]["false_generation_flush_count"], 0)

    def test_rejects_one_false_generation_flush(self) -> None:
        mutated = GOOD.replace(
            "ds4: GLM sync branch=indexed_resume",
            "ds4: CUDA persistent expert cache flushed (model load generation changed)\n"
            "ds4: GLM sync branch=indexed_resume",
        )
        self.assertEqual(score(mutated)["verdict"], "FAIL")

    def test_rejects_missing_cache_coverage(self) -> None:
        self.assertEqual(
            score(GOOD.replace("ds4: CUDA persistent expert cache enabled: 4110 slots x 9.28 MiB\n", ""))["verdict"],
            "FAIL",
        )

    def test_rejects_unfinished_resume(self) -> None:
        self.assertEqual(
            score(GOOD.replace("0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt done 3.500s\n", ""))["verdict"],
            "FAIL",
        )

    def test_rejects_missing_shutdown_marker(self) -> None:
        self.assertEqual(
            score(GOOD.replace("0807 15:10:07 ds4-server: shutdown requested, draining requests\n", ""))["verdict"],
            "FAIL",
        )

    def test_rejects_missing_or_duplicate_activation(self) -> None:
        self.assertEqual(score(GOOD.replace("ds4: CUDA stable model remap enabled generation=1\n", ""))["verdict"], "FAIL")
        duplicate = GOOD.replace(
            "ds4: CUDA stable model remap enabled generation=1\n",
            "ds4: CUDA stable model remap enabled generation=1\n" * 2,
        )
        self.assertEqual(score(duplicate)["verdict"], "FAIL")

    def test_rejects_unclean_child_exit(self) -> None:
        result = MODULE.score_text(
            GOOD,
            http_status=HTTP,
            response_text=RESPONSE,
            containment_rc=RC,
            containment_stdout=CONTAINMENT,
            mode="on",
            child_exit_text='{"shutdown_requested":true,"forced_kill":true,"exit_status":137}',
            safety_main_text=SAFETY,
            expected_binary_sha256=BINARY_SHA256,
            expected_environment_sha256=ENV_SHA256,
            model_identity_text=MODEL_IDENTITY,
            expected_model_sha256=MODEL_SHA256,
            expected_model_bytes=211075856448,
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_rejects_wrong_model_identity(self) -> None:
        result = MODULE.score_text(
            GOOD,
            http_status=HTTP,
            response_text=RESPONSE,
            containment_rc=RC,
            containment_stdout=CONTAINMENT,
            mode="on",
            child_exit_text='{"shutdown_requested":true,"forced_kill":false,"exit_status":0}',
            safety_main_text=SAFETY,
            expected_binary_sha256=BINARY_SHA256,
            expected_environment_sha256=ENV_SHA256,
            model_identity_text=MODEL_IDENTITY.replace(MODEL_SHA256, "0" * 64),
            expected_model_sha256=MODEL_SHA256,
            expected_model_bytes=211075856448,
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_summary_publication_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            MODULE.write_exclusive(path, "{}\n")
            with self.assertRaises(FileExistsError):
                MODULE.write_exclusive(path, "{}\n")

    def test_ignores_startup_and_post_shutdown_noise(self) -> None:
        noise = "ds4: CUDA persistent expert cache flushed (model load generation changed)\n"
        self.assertEqual(score(noise + GOOD + noise)["verdict"], "PASS")

    def test_rejects_unrelated_later_completion(self) -> None:
        bad = GOOD.replace(
            "ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1\n"
            "0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt done 3.500s\n",
            "ds4: GLM sync branch=indexed_resume pos=5044 chunk=22 logits=1\n"
            "0807 15:10:04 ds4-server: completion ctx=5044..5066:22 request failed\n"
            "0807 15:10:05 ds4-server: completion ctx=0..7:7 prompt start\n"
            "0807 15:10:06 ds4-server: completion ctx=0..7:7 prompt done 1.000s\n",
        )
        self.assertEqual(score(bad)["verdict"], "FAIL")

    def test_rejects_equal_context_tuple_response_collision(self) -> None:
        second = (
            "0807 15:10:06 ds4-server: completion ctx=5044..5066:22 prompt start\n"
            "0807 15:10:07 ds4-server: completion ctx=5044..5066:22 prompt done 1.000s\n"
        )
        bad = GOOD.replace(
            "0807 15:10:07 ds4-server: shutdown requested",
            second + "0807 15:10:08 ds4-server: shutdown requested",
        )
        self.assertEqual(score(bad)["verdict"], "FAIL")

    def test_rejects_fatal_after_completion(self) -> None:
        bad = GOOD.replace(
            "0807 15:10:07 ds4-server: shutdown requested",
            "ds4: CUDA GLM prefill failed\n0807 15:10:07 ds4-server: shutdown requested",
        )
        self.assertEqual(score(bad)["verdict"], "FAIL")

    def test_rejects_bad_http_response_or_containment(self) -> None:
        self.assertEqual(score(http="500\n")["verdict"], "FAIL")
        self.assertEqual(score(response="{}\n")["verdict"], "FAIL")
        self.assertEqual(score(rc="1\n")["verdict"], "FAIL")
        self.assertEqual(score(containment="SAFE_RUN_DONE rc=0 killed=yes dir=/tmp/x\n")["verdict"], "FAIL")

    def test_rejects_malformed_or_error_response_shapes(self) -> None:
        malformed = [
            '{"choices":[{"finish_reason":{},"text":null}],"usage":{"prompt_tokens":5066}}',
            '{"error":{"message":"failed"},"choices":[{"finish_reason":"error","text":""}],"usage":{"prompt_tokens":5066}}',
            '{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":5066}}',
            '{"choices":[{"finish_reason":"unsupported","text":""}],"usage":{"prompt_tokens":5066}}',
        ]
        for response in malformed:
            with self.subTest(response=response):
                self.assertEqual(score(response=response)["verdict"], "FAIL")

    def test_rejects_output_when_completion_token_count_is_zero(self) -> None:
        import json

        for unexpected in ("unexpected-output", " ", "\n"):
            with self.subTest(unexpected=repr(unexpected)):
                payload = json.loads(RESPONSE)
                payload["choices"][0]["text"] = unexpected
                self.assertEqual(
                    score(response=json.dumps(payload))["verdict"], "FAIL"
                )

    def test_rejects_canonical_fatal_markers_case_insensitively(self) -> None:
        for marker in (
            "CUDA_ERROR_OUT_OF_MEMORY",
            "cudaErrorMemoryAllocation",
            "CUDA runtime allocation failed",
            "CUDA runtime out of memory",
            "NV_ERR_NO_MEMORY",
            "FATAL ERROR",
            "FATAL CUDA userspace GPU/OOM evidence appeared",
            "oom-kill",
            "Out of memory: Killed process 123",
            "NVRM: Xid (PCI:0000:01:00): 31",
        ):
            with self.subTest(marker=marker):
                mutated = GOOD.replace(
                    "0807 15:10:07 ds4-server: shutdown requested",
                    f"{marker}\n0807 15:10:07 ds4-server: shutdown requested",
                )
                self.assertEqual(score(mutated)["verdict"], "FAIL")


if __name__ == "__main__":
    unittest.main()
