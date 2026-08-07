import json
import base64
import hashlib
import importlib.util
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
HARNESS = ROOT / "results/glm52-gates/harness/w7_resume_compiled_red_v1.sh"
TRACE_SCORER_PATH = ROOT / "scripts/83_score_w7_deployed_trace.py"


def _load_trace_scorer():
    spec = importlib.util.spec_from_file_location("w7_trace_scorer", TRACE_SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class W7CompiledRedHarnessTests(unittest.TestCase):
    def test_fixture_pool_binds_completion_parser_rendered_wire(self):
        pool = json.loads(
            (ROOT / "results/glm52-gates/harness/w7-production-fixture-pool-v1.json").read_text()
        )
        self.assertEqual(pool["schema"], "glm52-w7-production-fixture-pool-v2")
        self.assertEqual(
            pool["render_contract"],
            {
                "api": "/v1/completions",
                "context_tokens": 8192,
                "model": "default",
                "reasoning_effort": "high",
                "thinking": True,
                "system": "You are a helpful assistant",
                "oracle": "frozen-ds4-server-c-parser",
            },
        )
        primary = pool["variants"][0]
        self.assertEqual(primary["variant"], "primary-fixed")
        self.assertEqual(
            {
                "selected": primary["selected_tokens"],
                "common": primary["common_tokens"],
                "live": primary["live_tokens"],
                "prompt": primary["prompt_tokens"],
            },
            {"selected": 5044, "common": 5045, "live": 5055, "prompt": 5066},
        )

    def test_frozen_production_geometry_and_binary_are_available(self):
        completed = subprocess.run(
            ["/usr/bin/bash", str(HARNESS), "--self-test"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "W7_RED_SELFTEST_OK")

    def test_oracle_build_recipe_reproduces_frozen_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "oracle"
            completed = subprocess.run(
                [
                    str(ROOT / "scripts/82_build_w7_render_oracle.sh"),
                    "/tmp/glm52-w7-build1.ob4Q0O/src",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(
                subprocess.check_output(["sha256sum", str(output)], text=True).split()[0],
                "6bd6896581db71bdb76a9afdb59a9254b151ade22017e17f111fd3345fb5ad66",
            )

    def test_acceptance_is_resume_not_guard_bypass(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn("GLM sync start=5044 prompt=5066 suffix=22", source)
        self.assertIn("strict_guard_cold_restart", source)
        self.assertIn("readonly ENGINE_LOCK=/run/user/1000/ds4-engine.lock", source)
        self.assertIn("DS4_LOCK_EXPECTED_DEV_INO=$engine_lock_identity", source)
        assignments = [
            line for line in source.splitlines()
            if line.strip().startswith("DS4_GLM_RESUME_GUARD_OFF=")
        ]
        self.assertEqual(assignments, [])

    def test_deployed_trace_binds_exact_c_rendered_fixture(self):
        source = HARNESS.read_text(encoding="utf-8")
        self.assertIn('--trace "$out/request.trace"', source)
        self.assertIn('"thinking": True', source)
        self.assertIn('"reasoning_effort": "high"', source)
        self.assertIn('"trace_request_bytes_exact"', source)
        self.assertIn('"trace_rendered_bytes_exact"', source)
        self.assertIn('"trace_token_vectors_exact"', source)

    def test_runtime_scorer_and_tokenizer_dependencies_are_hash_pinned(self):
        source = HARNESS.read_text(encoding="utf-8")
        for name in (
            "TRACE_SCORER_SHA256",
            "TOKENIZER_SHA256",
            "TOKENIZER_INIT_SHA256",
            "TOKENIZER_NATIVE_SHA256",
        ):
            self.assertIn(f"readonly {name}=", source)
        for path, digest in (
            ("TRACE_SCORER", "TRACE_SCORER_SHA256"),
            ("TOKENIZER", "TOKENIZER_SHA256"),
            ("TOKENIZER_INIT", "TOKENIZER_INIT_SHA256"),
            ("TOKENIZER_NATIVE", "TOKENIZER_NATIVE_SHA256"),
        ):
            self.assertIn(f'"${path}" "${digest}"', source)
        self.assertIn("--validate-trace-result", source)

    def test_trace_result_contract_rejects_exit_schema_and_verdict_mutations(self):
        checks = {
            "trace_exactly_two_requests": True,
            "trace_request_ids_exact": True,
            "trace_request_bytes_exact": True,
            "trace_rendered_bytes_exact": True,
            "trace_token_vectors_exact": True,
        }
        observation = {
            "request_sha256": "1" * 64,
            "rendered_sha256": "2" * 64,
            "token_count": 1,
            "token_ids_sha256": "3" * 64,
        }
        valid = {
            "schema_version": 1,
            "checks": checks,
            "observed": [observation, observation],
            "error": None,
            "verdict": "PASS",
        }

        def validate(document, returncode=0):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
                json.dump(document, handle)
                handle.flush()
                return subprocess.run(
                    [str(HARNESS), "--validate-trace-result", str(returncode), handle.name],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )

        self.assertEqual(validate(valid).returncode, 0)
        mutations = []
        mutations.append((valid, 2))
        for key, value in (("verdict", "FAIL"), ("error", "forged"), ("schema_version", 2)):
            changed = json.loads(json.dumps(valid))
            changed[key] = value
            mutations.append((changed, 0))
        for invalid_version in (1.0, "1", True, None):
            changed = json.loads(json.dumps(valid))
            changed["schema_version"] = invalid_version
            mutations.append((changed, 0))
        extra = json.loads(json.dumps(valid))
        extra["unexpected"] = True
        mutations.append((extra, 0))
        false_check = json.loads(json.dumps(valid))
        false_check["checks"]["trace_request_bytes_exact"] = False
        mutations.append((false_check, 0))
        missing_check = json.loads(json.dumps(valid))
        del missing_check["checks"]["trace_request_ids_exact"]
        mutations.append((missing_check, 0))
        for document, returncode in mutations:
            with self.subTest(returncode=returncode, document=document):
                self.assertNotEqual(validate(document, returncode).returncode, 0)

    def test_execution_authority_rejects_harness_descendant(self):
        original = HARNESS.read_bytes()
        expected_sha256 = hashlib.sha256(original).hexdigest()
        candidate = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        valid = subprocess.run(
            [str(HARNESS), "--validate-execution-authority", expected_sha256, candidate],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)
        with tempfile.TemporaryDirectory() as directory:
            descendant = pathlib.Path(directory) / HARNESS.name
            descendant.write_bytes(original + b"\n# descendant mutation\n")
            descendant.chmod(0o700)
            rejected = subprocess.run(
                [str(descendant), "--validate-execution-authority", expected_sha256, candidate],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)

    def test_score_phase_uses_stable_scorer_descriptor_and_rechecks_dependencies(self):
        source = HARNESS.read_text(encoding="utf-8")
        score_phase = source[source.index("score_red() {"):source.index("if [[ ${1:-}")]
        self.assertIn('verify_runtime_dependencies', score_phase)
        self.assertIn('exec {trace_scorer_fd}<"$TRACE_SCORER"', score_phase)
        self.assertIn('/proc/$$/fd/$trace_scorer_fd', score_phase)
        self.assertGreaterEqual(score_phase.count('verify_runtime_dependencies'), 2)
        self.assertIn('W7_EXECUTED_HARNESS_SHA256', source)
        self.assertIn('W7_FROZEN_CANDIDATE_COMMIT', source)

    def test_self_test_rejects_runtime_dependency_path_substitution(self):
        source = HARNESS.read_text(encoding="utf-8")
        substitutions = {
            "TRACE_SCORER": "$REPO/scripts/83_score_w7_deployed_trace.py",
            "TOKENIZER": "/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json",
            "TOKENIZER_INIT": "$TOKENIZER_RUNTIME/tokenizers/__init__.py",
            "TOKENIZER_NATIVE": "$TOKENIZER_RUNTIME/tokenizers/tokenizers.abi3.so",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            replacement = root / "substituted"
            replacement.write_bytes(b"not the frozen dependency")
            replacement.chmod(0o700)
            for name, original in substitutions.items():
                mutated = source.replace(
                    f"readonly {name}={original}",
                    f"readonly {name}={replacement}",
                    1,
                )
                self.assertNotEqual(mutated, source, name)
                script = root / f"harness-{name}"
                script.write_text(mutated, encoding="utf-8")
                script.chmod(0o700)
                completed = subprocess.run(
                    [str(script), "--self-test"],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertNotEqual(completed.returncode, 0, name)

    def test_trace_scorer_rejects_equal_length_render_mutation(self):
        scorer = _load_trace_scorer()
        pool = json.loads(
            (ROOT / "results/glm52-gates/harness/w7-production-fixture-pool-v1.json").read_text()
        )
        stem = json.loads(
            (ROOT / "results/glm52-gates/harness/fixture-glm-long8.json").read_text()
        )["prompt"]
        primary = next(item for item in pool["variants"] if item["variant"] == "primary-fixed")
        payloads = [
            {"model": "default", "prompt": stem + pool["live"]["suffix_utf8"], "max_tokens": 0, "temperature": 0, "thinking": True, "reasoning_effort": "high"},
            {"model": "default", "prompt": stem + "\n\n[W7 primary fixed] Explain why a restored prefix must be rewound before this appended request.", "max_tokens": 0, "temperature": 0, "thinking": True, "reasoning_effort": "high"},
        ]
        requests = [json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode() for item in payloads]
        rendered = [
            base64.b64decode(pool["live"]["rendered_wire_utf8_b64"]),
            base64.b64decode(primary["rendered_wire_utf8_b64"]),
        ]

        def block(index, request, prompt):
            return (
                f"\n===== request {index} test =====\n".encode()
                + scorer.RAW_MARKER + request + b"\n"
                + scorer.RENDERED_MARKER + prompt + b"\n"
                + scorer.GENERATED_MARKER + f"\n===== end request {index} =====\n".encode()
            )

        good_trace = block(1, requests[0], rendered[0]) + block(2, requests[1], rendered[1])
        args = (
            pool,
            requests[0],
            requests[1],
            pathlib.Path("/home/dsv4/ds4-project/tokenizers/glm52-b4734de4/tokenizer.json"),
            pathlib.Path("/home/bmarti44/.cache/glm52-w3-tokenizer-runtime-0.22.2"),
        )
        self.assertEqual(scorer.score_trace(good_trace, *args)["verdict"], "PASS")
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            fake_tokenizer = root / "tokenizer.json"
            fake_tokenizer.write_text("{}", encoding="utf-8")
            self.assertEqual(
                scorer.score_trace(good_trace, pool, requests[0], requests[1], fake_tokenizer, args[4])["verdict"],
                "FAIL",
            )
            fake_runtime = root / "runtime"
            (fake_runtime / "tokenizers").mkdir(parents=True)
            shutil.copyfile(args[4] / "tokenizers/__init__.py", fake_runtime / "tokenizers/__init__.py")
            (fake_runtime / "tokenizers/tokenizers.abi3.so").write_bytes(b"mutated native runtime")
            self.assertEqual(
                scorer.score_trace(good_trace, pool, requests[0], requests[1], args[3], fake_runtime)["verdict"],
                "FAIL",
            )
        mutated = bytes([rendered[1][0] ^ 1]) + rendered[1][1:]
        bad_trace = block(1, requests[0], rendered[0]) + block(2, requests[1], mutated)
        result = scorer.score_trace(bad_trace, *args)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertFalse(result["checks"]["trace_rendered_bytes_exact"])
        malformed = {
            "missing": block(1, requests[0], rendered[0]),
            "duplicate": good_trace + block(3, requests[1], rendered[1]),
            "truncated": good_trace[:-24],
            "swapped": block(1, requests[1], rendered[1]) + block(2, requests[0], rendered[0]),
            "ambiguous-marker": good_trace.replace(
                scorer.GENERATED_MARKER,
                scorer.GENERATED_MARKER + scorer.RENDERED_MARKER,
                1,
            ),
            "leading-whitespace": b" \t\r\n" + good_trace,
            "leading-nul": b"\x00" + good_trace,
            "duplicate-request-id": good_trace.replace(
                b"request 2", b"request 1"
            ),
            "reversed-request-ids": good_trace.replace(
                b"request 1", b"request 9"
            ).replace(b"request 2", b"request 1").replace(b"request 9", b"request 2"),
        }
        for name, trace in malformed.items():
            with self.subTest(name=name):
                self.assertEqual(scorer.score_trace(trace, *args)["verdict"], "FAIL")

    def test_preregistered_plan_authorizes_compiled_red(self):
        plan = json.loads(
            (ROOT / "results/glm52-gates/W7-resume-correctness-plan-v8.json").read_text()
        )
        self.assertEqual(plan["status"], "C_PARSER_FIXTURE_CORRECTION_AUTHORIZED_NOT_EXECUTED")
        self.assertEqual(
            plan["compiled_red_classification"]["geometry"],
            {"selected": 5044, "common": 5045, "live": 5055, "prompt": 5066},
        )


if __name__ == "__main__":
    unittest.main()
