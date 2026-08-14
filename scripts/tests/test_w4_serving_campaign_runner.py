#!/usr/bin/env python3

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "w4_serving_runner", ROOT / "scripts/102_run_w4_serving_campaign.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class W4ServingContainmentTest(unittest.TestCase):
    def test_candidate_check_uses_retained_bytes_with_synthetic_self_path(self) -> None:
        candidate = "3" * 40
        retained = b"reviewed retained runner bytes"
        head = mock.Mock(stdout=candidate + "\n")
        clean = mock.Mock(stdout="")
        prior_file = RUNNER.__file__
        prior_retained = getattr(RUNNER, "_W4_EXECUTED_RUNNER_BYTES", None)
        RUNNER.__file__ = "/w4/frozen/scripts/102_run_w4_serving_campaign.py"
        RUNNER._W4_EXECUTED_RUNNER_BYTES = retained
        try:
            with mock.patch.object(RUNNER.subprocess, "run", side_effect=(head, clean)), \
                 mock.patch.object(RUNNER, "git_bytes", return_value=retained):
                RUNNER.verify_candidate(candidate)
        finally:
            RUNNER.__file__ = prior_file
            if prior_retained is None:
                del RUNNER._W4_EXECUTED_RUNNER_BYTES
            else:
                RUNNER._W4_EXECUTED_RUNNER_BYTES = prior_retained

    def test_live_arm_launch_uses_retained_runner_cgroup_safe_and_base(self) -> None:
        campaign_source = inspect.getsource(RUNNER._campaign)
        self.assertIn("retained_arm_dependencies", campaign_source)
        self.assertIn("contained_arm_command", campaign_source)
        self.assertNotIn('str(Path(__file__))', campaign_source)
        self.assertNotIn('command = [str(CGROUP)', campaign_source)

    def test_retained_arm_dependencies_ignore_path_swaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                "runner": root / "runner.py",
                "base": root / "base.py",
                "cgroup": root / "cgroup.sh",
                "safe": root / "safe.sh",
            }
            originals = {name: f"reviewed-{name}\n".encode() for name in paths}
            for name, path in paths.items():
                path.write_bytes(originals[name])
            hashes = {name: hashlib.sha256(payload).hexdigest()
                      for name, payload in originals.items()}
            with mock.patch.object(RUNNER, "RUNNER_PATH", paths["runner"], create=True), \
                 mock.patch.object(RUNNER, "BASE_PATH", paths["base"]), \
                 mock.patch.object(RUNNER, "CGROUP", paths["cgroup"]), \
                 mock.patch.object(RUNNER, "SAFE", paths["safe"]), \
                 mock.patch.object(RUNNER, "BASE_SHA256", hashes["base"]), \
                 mock.patch.object(RUNNER, "CGROUP_SHA256", hashes["cgroup"]), \
                 mock.patch.object(RUNNER, "SAFE_SHA256", hashes["safe"]), \
                 RUNNER.retained_arm_dependencies() as retained:
                for name, path in paths.items():
                    path.unlink()
                    path.write_bytes(f"forged-{name}\n".encode())
                self.assertEqual(
                    {name: Path(retained[name + "_path"]).read_bytes()
                     for name in paths}, originals)
                command = RUNNER.contained_arm_command(
                    retained, "w4-test", ["--driver", "off"])
                rendered = "\0".join(command)
                self.assertNotIn(str(paths["runner"]), rendered)
                self.assertNotIn(str(paths["cgroup"]), rendered)
                self.assertNotIn(str(paths["safe"]), rendered)
                self.assertIn(retained["runner_path"], rendered)
                self.assertIn(retained["base_path"], rendered)
                self.assertIn(retained["cgroup_path"], rendered)

    def test_safe_run_candidate_directory_contains_named_binary(self) -> None:
        self.assertEqual(RUNNER.BIN.name, "ds4-server")
        self.assertEqual(RUNNER.CANDIDATE_SRC, RUNNER.BIN.parent)
        self.assertTrue((RUNNER.CANDIDATE_SRC / "ds4-server").is_file())

    def test_containment_forwards_exact_topk_flag(self) -> None:
        source = (ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh").read_text()
        self.assertIn("  DS4_CUDA_TOPK2048_CUB \\\n", source)

    def test_arm_environments_differ_only_by_topk_flag_and_logit_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            off, _ = RUNNER.environment_for_arm("off", parent / "off", "/proc/123/fd/9", "1:2")
            on, _ = RUNNER.environment_for_arm("on", parent / "on", "/proc/123/fd/9", "1:2")
        ignored = {
            "DS4_CUDA_TOPK2048_CUB", "DS4_GLM_LOGIT_DUMP",
            "GLM_SAFE_EXPECTED_ENV_SHA256", "GLM_SAFE_PROVENANCE_ENV_ALLOWLIST",
            "GLM_SAFE_FINAL_ARTIFACTS",
        }
        self.assertEqual({k: v for k, v in off.items() if k not in ignored},
                         {k: v for k, v in on.items() if k not in ignored})
        self.assertNotIn("DS4_CUDA_TOPK2048_CUB", off)
        self.assertEqual(on["DS4_CUDA_TOPK2048_CUB"], "1")
        self.assertEqual(off["DS4_CUDA_STABLE_MODEL_REMAP"], "1")
        self.assertEqual(on["DS4_CUDA_STABLE_MODEL_REMAP"], "1")
        off_measured = {name: off[name] for name in
                        off["GLM_SAFE_PROVENANCE_ENV_ALLOWLIST"].split(",")}
        on_measured = {name: on[name] for name in
                       on["GLM_SAFE_PROVENANCE_ENV_ALLOWLIST"].split(",")}
        self.assertEqual(RUNNER.validate_environment_artifact(
            "off", Path(directory) / "off", off_measured),
            off["GLM_SAFE_EXPECTED_ENV_SHA256"])
        self.assertEqual(RUNNER.validate_environment_artifact(
            "on", Path(directory) / "on", on_measured),
            on["GLM_SAFE_EXPECTED_ENV_SHA256"])
        mutated = dict(on_measured)
        mutated["DS4_CUDA_FETCH_THREADS"] = "7"
        with self.assertRaises(RUNNER.CampaignError):
            RUNNER.validate_environment_artifact("on", Path(directory) / "on", mutated)

    def test_request_is_deterministic_and_non_generating(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            first_sha = RUNNER.make_request(first)
            second_sha = RUNNER.make_request(second)
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            doc = json.loads(first_bytes)
            tokenization = RUNNER.load_scorer(RUNNER.SCORER.read_bytes()).independent_tokenization(first)
        self.assertEqual(first_sha, second_sha)
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(doc["max_tokens"], 0)
        self.assertFalse(doc["stream"])
        self.assertGreater(len(doc["prompt"]), 90_000)
        self.assertEqual(tokenization["prompt_tokens"], 19_783)

    def test_schedule_is_deterministic_and_domain_separated(self) -> None:
        seed = hashlib.sha256(b"post-freeze randomness").hexdigest()
        self.assertEqual(RUNNER.derive_schedules(seed), RUNNER.derive_schedules(seed))
        self.assertEqual(len(RUNNER.derive_schedules(seed)), 5)
        self.assertLess(RUNNER.drand_publication_time(6_359_296), 1_786_210_399)

    def test_github_push_time_is_external_randomness_floor(self) -> None:
        candidate = "3" * 40
        response = mock.MagicMock()
        response.read.return_value = json.dumps([{
            "id": "event-1", "type": "PushEvent", "created_at": "2026-08-08T17:45:25Z",
            "payload": {"head": candidate,
                        "ref": "refs/heads/glm52-rung0-io-submission"},
        }]).encode()
        response.__enter__.return_value = response
        opener = mock.MagicMock()
        opener.open.return_value = response
        with mock.patch.object(RUNNER.urllib.request, "build_opener", return_value=opener):
            receipt = RUNNER.fetch_publication_receipt(candidate)
        self.assertEqual(receipt["candidate_hash"], candidate)
        self.assertEqual(receipt["created_at_unix"], 1_786_211_125)

    def test_randomness_requires_first_round_strictly_after_publication(self) -> None:
        candidate = "3" * 40
        publication = {"candidate_hash": candidate, "created_at_unix": 1_786_212_006}
        first = 6_359_367
        base = {
            "freeze_floor_round": RUNNER.DRAND_FREEZE_FLOOR_ROUND,
            "randomness": "4" * 64,
            "signature": "5" * 192,
            "previous_signature": "6" * 192,
            "frozen_gate_commit": candidate,
            "relay_agreement": ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"],
        }
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(RUNNER, "_sealed_bls_verify", return_value=True):
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps({**base, "round": first}))
            self.assertEqual(RUNNER.verify_randomness(
                receipt, candidate, publication)[0], "4" * 64)
            for wrong_round in (first - 1, first + 1, first + 100):
                receipt.write_text(json.dumps({**base, "round": wrong_round}))
                with self.subTest(round=wrong_round), self.assertRaisesRegex(
                        RUNNER.CampaignError, "first eligible"):
                    RUNNER.verify_randomness(receipt, candidate, publication)

    def test_randomness_verification_rejects_mutated_bundle(self) -> None:
        candidate = "3" * 40
        publication = {"candidate_hash": candidate, "created_at_unix": 1_786_212_006}
        doc = {
            "round": 6_359_367,
            "freeze_floor_round": RUNNER.DRAND_FREEZE_FLOOR_ROUND,
            "randomness": "4" * 64,
            "signature": "5" * 192,
            "previous_signature": "6" * 192,
            "frozen_gate_commit": candidate,
            "relay_agreement": ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"],
        }
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "verifier.mjs"
            bundle.write_text("forged verifier")
            receipt = Path(directory) / "receipt.json"
            receipt.write_text(json.dumps(doc))
            with mock.patch.object(RUNNER, "DRAND_BUNDLE", bundle), \
                 self.assertRaisesRegex(RUNNER.CampaignError, "verifier dependency"):
                RUNNER.verify_randomness(receipt, candidate, publication)

    def test_randomness_executes_only_retained_sealed_verifier_bytes(self) -> None:
        candidate = "3" * 40
        publication = {"candidate_hash": candidate, "created_at_unix": 1_786_212_006}
        doc = {
            "round": 6_359_367,
            "freeze_floor_round": RUNNER.DRAND_FREEZE_FLOOR_ROUND,
            "randomness": "4" * 64,
            "signature": "5" * 192,
            "previous_signature": "6" * 192,
            "frozen_gate_commit": candidate,
            "relay_agreement": ["api.drand.sh", "api2.drand.sh", "api3.drand.sh"],
        }
        with mock.patch.object(RUNNER.subprocess, "run",
                               side_effect=AssertionError("unsealed subprocess used")), \
             mock.patch.object(RUNNER, "_sealed_bls_verify", return_value=True,
                               create=True) as sealed:
            seed, _, _ = RUNNER.verify_randomness_bytes(
                json.dumps(doc).encode(), candidate, publication)
        self.assertEqual(seed, "4" * 64)
        sealed.assert_called_once()

    def test_publication_fetch_ignores_ambient_proxy_and_ca_overrides(self) -> None:
        candidate = "3" * 40
        response = mock.MagicMock()
        response.read.return_value = json.dumps([{
            "id": "event-1", "type": "PushEvent", "created_at": "2026-08-08T17:45:25Z",
            "payload": {"head": candidate,
                        "ref": "refs/heads/glm52-rung0-io-submission"},
        }]).encode()
        response.__enter__.return_value = response
        opener = mock.MagicMock()
        opener.open.return_value = response
        tls_context = mock.MagicMock()
        hostile = {"HTTPS_PROXY": "https://127.0.0.1:1",
                   "SSL_CERT_FILE": "/tmp/attacker-ca.pem",
                   "SSL_CERT_DIR": "/tmp/attacker-ca"}
        with mock.patch.dict(os.environ, hostile), \
             mock.patch.object(RUNNER.urllib.request, "urlopen",
                               side_effect=AssertionError("ambient opener used")), \
             mock.patch.object(RUNNER.ssl, "SSLContext",
                               return_value=tls_context) as create_context, \
             mock.patch.object(RUNNER.urllib.request, "build_opener", return_value=opener) as build:
            receipt = RUNNER.fetch_publication_receipt(candidate)
        self.assertEqual(receipt["candidate_hash"], candidate)
        opener.open.assert_called_once()
        create_context.assert_called_once_with(RUNNER.ssl.PROTOCOL_TLS_CLIENT)
        tls_context.load_verify_locations.assert_called_once()
        self.assertIn("BEGIN CERTIFICATE",
                      tls_context.load_verify_locations.call_args.kwargs["cadata"])
        handlers = build.call_args.args
        self.assertTrue(any(isinstance(handler, RUNNER.urllib.request.ProxyHandler)
                            and handler.proxies == {} for handler in handlers))
        self.assertTrue(any(isinstance(handler, RUNNER.urllib.request.HTTPSHandler)
                            for handler in handlers))

    def test_sync_trace_requires_full_novel_prefill(self) -> None:
        first = ("ds4: GLM sync start=0 prompt=19772 suffix=19772 checkpoint=0 "
                 "dense_len=0 ctx_cap=8192 dense_fit=0 resume_min=4 dense_gap=0 "
                 "indexed_keep=0 indexed_batch=1 batch_ffn=1")
        second = ("ds4: GLM sync start=19772 prompt=19783 suffix=11 checkpoint=19772 "
                  "dense_len=0 ctx_cap=8192 dense_fit=0 resume_min=4 dense_gap=1 "
                  "indexed_keep=1 indexed_batch=1 batch_ffn=1")
        valid = first + "\n" + second
        RUNNER.validate_novel_sync_trace(valid, 19_783)
        for mutation in (
            valid.replace("start=0", "start=1"),
            valid.replace("start=19772 prompt=19783", "start=19773 prompt=19783"),
            valid.replace("suffix=11", "suffix=10"),
            valid.replace("checkpoint=19772", "checkpoint=19771"),
            valid + "\n" + second,
            valid + "\n" + second + " hidden=1",
        ):
            with self.subTest(mutation=mutation[:60]), self.assertRaises(RUNNER.CampaignError):
                RUNNER.validate_novel_sync_trace(mutation, 19_783)

    def test_arm_replay_is_pure_self_contained_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "b0-p0-off-0123456789ab"
            safety = out / "safety"
            safety.mkdir(parents=True)
            environment = RUNNER.measured_environment(
                "off", out, "/proc/123/fd/9", "1:2")
            response = {"choices": [{"text": "", "finish_reason": "length"}],
                        "usage": {"prompt_tokens": 19_783, "completion_tokens": 0,
                                  "total_tokens": 19_783,
                                  "prompt_tokens_details": {
                                      "cached_tokens": 0,
                                      "cache_write_tokens": 19_783}}}
            observation = {"request_start_ns": 1, "response_complete_ns": 2,
                           "semantic": response,
                           "response_semantic_sha256": "0" * 64}
            files = {
                "server.log": ("ds4-server: listening on 127.0.0.1\n"
                               "ds4: GLM sync start=0 prompt=19772 suffix=19772 checkpoint=0 "
                               "dense_len=0 ctx_cap=8192 dense_fit=0 resume_min=4 dense_gap=0 "
                               "indexed_keep=0 indexed_batch=1 batch_ffn=1\n"
                               "ds4: GLM sync start=19772 prompt=19783 suffix=11 "
                               "checkpoint=19772 dense_len=0 ctx_cap=8192 dense_fit=0 "
                               "resume_min=4 dense_gap=1 indexed_keep=1 indexed_batch=1 "
                               "batch_ffn=1\n"
                               "ds4-server: shutdown requested\n").encode(),
                "response.json": json.dumps(response).encode(),
                "observation.json": json.dumps(observation).encode(),
                "environment.json": json.dumps(environment).encode(),
                "child-exit.json": b'{"exit_status":0}',
                "driver-lineage.json": json.dumps({
                    "runner_sha256": hashlib.sha256(RUNNER.RUNNER_PATH.read_bytes()).hexdigest(),
                    "base_runner_sha256": RUNNER.BASE_SHA256,
                    "cgroup_sha256": RUNNER.CGROUP_SHA256,
                    "safe_run_sha256": RUNNER.SAFE_SHA256,
                }, sort_keys=True, separators=(",", ":")).encode() + b"\n",
            }
            for name, payload in files.items():
                (out / name).write_bytes(payload)
            logit1 = out / "logits.sync1.start0.prompt19772.suffix19772"
            logit2 = out / "logits.sync2.start19772.prompt19783.suffix11"
            logit1.write_bytes(b"\0" * RUNNER.LOGIT_BYTES)
            logit2.write_bytes(b"\1" * RUNNER.LOGIT_BYTES)
            env_sha = RUNNER.validate_environment_artifact("off", out, environment)
            markers = []
            for name, payload in files.items():
                metadata = (out / name).stat()
                markers.append(
                    f"final_artifact_verified path={out / name} "
                    f"sha256={hashlib.sha256(payload).hexdigest()} "
                    f"device_inode={metadata.st_dev}:{metadata.st_ino}:{metadata.st_size}")
            main = (f"executed_environment_allowlist={','.join(sorted(environment))} "
                    f"executed_environment_sha256={env_sha}\n" + "\n".join(markers) +
                    "\ncgroup_final memory_current_bytes=1 swap_current_bytes=0 "
                    "events=low 0,high 0,max 0,oom 0,oom_kill 0,oom_group_kill 0,\n"
                    "wrapper and descendant checks clean\n").encode()
            samples = b"mem_avail_kb=20971520 cgroup_swap_current_bytes=0\n"
            kernel = b"no driver faults\n"
            for name, payload in (("main.log", main), ("samples.log", samples),
                                  ("kernel.log", kernel)):
                (safety / name).write_bytes(payload)
            done = ("SAFE_RUN_DONE rc=0 killed=no "
                    "dir=/home/bmarti44/.local/state/glm52-crashlog/nonexistent "
                    f"main_sha256={hashlib.sha256(main).hexdigest()} "
                    f"samples_sha256={hashlib.sha256(samples).hexdigest()} "
                    f"kernel_sha256={hashlib.sha256(kernel).hexdigest()}\n")
            with mock.patch.object(RUNNER.BASE, "server_pids", return_value=[999]):
                first = RUNNER.parse_arm("off", 0, 0, out, 0, done, "4" * 64,
                                         "5" * 64, 19_783, None, True)
                second = RUNNER.parse_arm("off", 0, 0, out, 0, done, "4" * 64,
                                          "5" * 64, 19_783, None, True)
                wrong = out / "logits.sync2.start19772.prompt19783.suffix10"
                logit2.rename(wrong)
                with self.assertRaisesRegex(RUNNER.CampaignError,
                                            "does not match sync trace"):
                    RUNNER.parse_arm("off", 0, 0, out, 0, done, "4" * 64,
                                     "5" * 64, 19_783, None, True)
                wrong.rename(logit2)
                duplicate = out / "logits.sync3.start19772.prompt19783.suffix11"
                duplicate.write_bytes(b"\2" * RUNNER.LOGIT_BYTES)
                with self.assertRaisesRegex(RUNNER.CampaignError,
                                            "does not match sync trace"):
                    RUNNER.parse_arm("off", 0, 0, out, 0, done, "4" * 64,
                                     "5" * 64, 19_783, None, True)
                duplicate.unlink()
                shadow = out / "logits.shadow"
                shadow.write_bytes(b"\2" * RUNNER.LOGIT_BYTES)
                with self.assertRaisesRegex(RUNNER.CampaignError,
                                            "logit artifact closure"):
                    RUNNER.parse_arm("off", 0, 0, out, 0, done, "4" * 64,
                                     "5" * 64, 19_783, None, True)
                shadow.unlink()
                noncanonical = out / "logits.sync01.start0.prompt19772.suffix19772"
                logit1.rename(noncanonical)
                with self.assertRaisesRegex(RUNNER.CampaignError,
                                            "logit artifact closure"):
                    RUNNER.parse_arm("off", 0, 0, out, 0, done, "4" * 64,
                                     "5" * 64, 19_783, None, True)
                noncanonical.rename(logit1)
        self.assertEqual(first, second)
        self.assertEqual(first["safety"]["surviving_descendants"], 0)
        expected_final = hashlib.sha256(b"\1" * RUNNER.LOGIT_BYTES).hexdigest()
        expected_sequence = hashlib.sha256(json.dumps([
            ("logits.sync1.start0.prompt19772.suffix19772",
             hashlib.sha256(b"\0" * RUNNER.LOGIT_BYTES).hexdigest(), RUNNER.LOGIT_BYTES),
            ("logits.sync2.start19772.prompt19783.suffix11",
             expected_final, RUNNER.LOGIT_BYTES),
        ], separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(first["final_logits_sha256"], expected_final)
        self.assertEqual(first["logit_sequence_sha256"], expected_sequence)

    def test_signal_handlers_are_restored_after_preflight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(RUNNER, "OUT_ROOT", Path(directory)), \
             mock.patch.object(RUNNER, "user_systemd_available", return_value=False), \
             mock.patch.object(RUNNER.BASE, "install_campaign_signal_handlers",
                               return_value={}) as installed, \
             mock.patch.object(RUNNER.BASE, "restore_campaign_signal_handlers") as restored:
            with self.assertRaises(RUNNER.CampaignError):
                RUNNER.campaign("0" * 40, Path("unused"))
            installed.assert_called_once_with()
            restored.assert_called_once_with({})
            RUNNER.BASE._ACTIVE_ATTEMPT = None
            RUNNER.BASE._ACTIVE_CANDIDATE = None

    def test_campaign_rejects_dead_user_manager_before_large_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(RUNNER, "OUT_ROOT", Path(directory)), \
             mock.patch.object(RUNNER, "user_systemd_available", return_value=False), \
             mock.patch.object(RUNNER, "verify_dependencies",
                               side_effect=AssertionError("must not hash dependencies")):
            error = RUNNER.CampaignError("user-systemd containment is unavailable")
            with self.assertRaisesRegex(RUNNER.CampaignError,
                                        "user-systemd containment is unavailable"):
                RUNNER.campaign("0" * 40, Path("unused-receipt.json"))
            RUNNER.finalize_failure(error)
            attempts = list(Path(directory).glob("attempt-*"))
            self.assertEqual(len(attempts), 1)
            self.assertTrue((attempts[0] / "manifest.json").is_file())
            RUNNER.BASE._ACTIVE_ATTEMPT = None
            RUNNER.BASE._ACTIVE_CANDIDATE = None

    def test_failure_finalizer_emits_w4_bound_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            RUNNER.BASE._ACTIVE_ATTEMPT = attempt
            RUNNER.BASE._ACTIVE_CANDIDATE = "1" * 40
            try:
                RUNNER.finalize_failure(RuntimeError("synthetic failure"))
                manifest = json.loads((attempt / "manifest.json").read_bytes())
            finally:
                RUNNER.BASE._ACTIVE_ATTEMPT = None
                RUNNER.BASE._ACTIVE_CANDIDATE = None
        self.assertEqual(manifest["schema"], "glm52-w4-serving-campaign-failure-v1")
        self.assertEqual(manifest["scorer_sha256"], RUNNER.SCORER_SHA256)
        self.assertEqual(manifest["binary_sha256"], RUNNER.BINARY_SHA256)

    def test_failure_after_provisional_pass_displaces_and_closes_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            (attempt / "raw.jsonl").write_bytes(b"{}\n")
            (attempt / "summary.json").write_text('{"verdict":"PASS"}\n')
            (attempt / "manifest.json").write_text('{"verdict":"PASS"}\n')
            (attempt / "retained.log").write_text("host observation\n")
            RUNNER.BASE._ACTIVE_ATTEMPT = attempt
            RUNNER.BASE._ACTIVE_CANDIDATE = "2" * 40
            try:
                RUNNER.finalize_failure(RuntimeError("replay failed"))
                manifest = json.loads((attempt / "manifest.json").read_bytes())
            finally:
                RUNNER.BASE._ACTIVE_ATTEMPT = None
                RUNNER.BASE._ACTIVE_CANDIDATE = None
        self.assertEqual(manifest["verdict"], "FAIL")
        self.assertIn("manifest.pre-failure.json", manifest["artifacts"])
        self.assertIn("summary.pre-finalization.json", manifest["artifacts"])
        self.assertIn("retained.log", manifest["artifacts"])

    def test_symlink_failure_preserves_terminal_authority_without_following(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            (attempt / "raw.jsonl").write_bytes(b"{}\n")
            (attempt / "summary.json").write_text('{"verdict":"PASS"}\n')
            (attempt / "manifest.json").write_text('{"verdict":"PASS"}\n')
            (attempt / "target").write_text("must not be read")
            (attempt / "hostile-link").symlink_to(attempt / "target")
            RUNNER.BASE._ACTIVE_ATTEMPT = attempt
            RUNNER.BASE._ACTIVE_CANDIDATE = "2" * 40
            try:
                RUNNER.finalize_failure(RuntimeError("symlink mutation"))
                manifest = json.loads((attempt / "manifest.json").read_bytes())
            finally:
                RUNNER.BASE._ACTIVE_ATTEMPT = None
                RUNNER.BASE._ACTIVE_CANDIDATE = None
        self.assertEqual(manifest["verdict"], "FAIL")
        self.assertEqual(manifest["rejected_symlinks"], ["hostile-link"])
        self.assertNotIn("hostile-link", manifest["artifacts"])

    def test_failure_finalizer_records_concurrent_unstable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            unstable = attempt / "unstable.log"
            unstable.write_text("racing evidence")
            original = RUNNER.BASE.read_stable
            def injected(path):
                if path == unstable:
                    raise OSError("regular-to-symlink race")
                return original(path)
            RUNNER.BASE._ACTIVE_ATTEMPT = attempt
            RUNNER.BASE._ACTIVE_CANDIDATE = "2" * 40
            try:
                with mock.patch.object(RUNNER.BASE, "read_stable", side_effect=injected):
                    RUNNER.finalize_failure(RuntimeError("race mutation"))
                manifest = json.loads((attempt / "manifest.json").read_bytes())
            finally:
                RUNNER.BASE._ACTIVE_ATTEMPT = None
                RUNNER.BASE._ACTIVE_CANDIDATE = None
        self.assertEqual(manifest["verdict"], "FAIL")
        self.assertEqual(manifest["rejected_unstable_paths"], ["unstable.log"])

    def test_failure_finalizer_survives_reserved_path_instability(self) -> None:
        for reserved in ("raw.jsonl", "summary.json"):
            with self.subTest(reserved=reserved), tempfile.TemporaryDirectory() as directory:
                attempt = Path(directory)
                (attempt / reserved).write_text("reserved evidence\n")
                original = RUNNER.BASE.read_stable
                def injected(path):
                    if path == attempt / reserved:
                        raise OSError("reserved path race")
                    return original(path)
                RUNNER.BASE._ACTIVE_ATTEMPT = attempt
                RUNNER.BASE._ACTIVE_CANDIDATE = "2" * 40
                caught = None
                manifest = None
                try:
                    with mock.patch.object(RUNNER.BASE, "read_stable", side_effect=injected):
                        try:
                            RUNNER.finalize_failure(RuntimeError("reserved mutation"))
                        except Exception as error:
                            caught = error
                    if (attempt / "manifest.json").is_file():
                        manifest = json.loads((attempt / "manifest.json").read_bytes())
                finally:
                    RUNNER.BASE._ACTIVE_ATTEMPT = None
                    RUNNER.BASE._ACTIVE_CANDIDATE = None
                self.assertIsNone(caught)
                self.assertIsNotNone(manifest)
                self.assertEqual(manifest["verdict"], "FAIL")
                self.assertIn(reserved, manifest["rejected_unstable_paths"])

    def test_failure_finalizer_reconciles_postread_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            raced = attempt / "raced.log"
            raced.write_text("reviewed evidence\n")
            original = RUNNER.BASE.read_stable
            replaced = False
            def injected(path):
                nonlocal replaced
                result = original(path)
                if path == raced and not replaced:
                    replaced = True
                    path.unlink()
                    path.write_text("replacement evidence\n")
                return result
            RUNNER.BASE._ACTIVE_ATTEMPT = attempt
            RUNNER.BASE._ACTIVE_CANDIDATE = "2" * 40
            try:
                with mock.patch.object(RUNNER.BASE, "read_stable", side_effect=injected):
                    RUNNER.finalize_failure(RuntimeError("post-read mutation"))
                manifest = json.loads((attempt / "manifest.json").read_bytes())
            finally:
                RUNNER.BASE._ACTIVE_ATTEMPT = None
                RUNNER.BASE._ACTIVE_CANDIDATE = None
        self.assertEqual(manifest["verdict"], "FAIL")
        self.assertNotIn("raced.log", manifest["artifacts"])
        self.assertIn("raced.log", manifest["rejected_unstable_paths"])


if __name__ == "__main__":
    unittest.main()
