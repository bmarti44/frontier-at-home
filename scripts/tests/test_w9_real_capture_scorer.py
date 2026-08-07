from __future__ import annotations

import array
import importlib.util
import json
import pathlib
import re
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/91_score_w9_real_capture.py"
HARNESS = ROOT / "results/glm52-gates/harness/w9_real_capture_v1.sh"
SPEC = importlib.util.spec_from_file_location("w9_scorer", SCORER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class W9RealCaptureScorerTests(unittest.TestCase):
    def _paired_fixture(self, root: pathlib.Path) -> pathlib.Path:
        binary = "b" * 64
        model = "c" * 64
        tokenizer = "d" * 64
        configuration = "e" * 64
        prompt = b"matched prompt"
        prompt_hash = MODULE.hashlib.sha256(prompt).hexdigest()
        arms = {}
        for arm_name in MODULE.ARMS:
            arm = root / arm_name
            arm.mkdir()
            (arm / "prompt.txt").write_bytes(prompt)
            (arm / "cli.stdout").write_bytes(b"")
            logit = arm / "logits.sync0.start0.prompt8192.suffix8192"
            logit.write_bytes(array.array("f", [0.0] * MODULE.LOGIT_COUNT).tobytes())
            stderr = f"ds4: prefill logits dumped to {logit}\n"
            if arm_name == "on":
                stderr += "ds4: W9 real capture complete rows=8192 query_rows=128 layers=1\n"
            (arm / "cli.stderr").write_text(stderr, encoding="utf-8")
            (arm / "containment.rc").write_text("0\n", encoding="utf-8")
            (arm / "containment.stdout").write_text(
                "SAFE_RUN_DONE rc=0 killed=no dir=/tmp/safe\n", encoding="utf-8")
            (arm / "containment.stderr").write_bytes(b"")
            safety = arm / "safety"
            safety.mkdir()
            (safety / "samples.log").write_text(
                "mem_avail_kb=30000000 eng_rss_kb=1 cgroup_current_bytes=1 "
                "cgroup_peak_bytes=1 cgroup_swap_current_bytes=0\n", encoding="utf-8")
            (safety / "kernel.log").write_bytes(b"")
            (safety / "main.log").write_text(
                "executed_candidate_verified pid=1 start_ticks=1 path=/candidate "
                f"executed_binary_sha256={binary} device_inode=1:2 \n"
                "wrapper and descendant checks clean\n"
                "cgroup_final current_bytes=0 peak_bytes=1 swap_current_bytes=0 "
                "events=low 0,high 0,max 0,oom 0,oom_kill 0,oom_group_kill 0,\n",
                encoding="utf-8")
            if arm_name == "on":
                capture = arm / "capture"
                capture.mkdir()
                (capture / "kv.f32").write_bytes(array.array("f", [0.0] * 4).tobytes())
                (capture / "query.f32").write_bytes(array.array("f", [0.0] * 4).tobytes())
                counts = array.array("I", [2048] * 128)
                (capture / "selected-count.u32").write_bytes(counts.tobytes())
                row = array.array("I", [0] + [8193] * 2047).tobytes()
                with (capture / "selected.u32").open("wb") as handle:
                    for _ in counts:
                        handle.write(row)
                metadata = {
                    "schema": "glm52-w9-real-capture-v1", "layers": [0],
                    "kv_rows_per_layer": 8192, "kv_width": 512,
                    "query_rows_per_layer": 128, "query_heads": 64,
                    "query_width": 512, "selected_capacity": 2048,
                    "sample_position_start": 0, "sample_position_stride": 64,
                    "selected_padding_sentinel": 8193,
                    "storage_padding_sentinel": 0xFFFFFFFF,
                    "artifacts": MODULE.CAPTURE_SIZES,
                    "dtype": {"kv": "f32", "query": "f32", "selected": "u32"},
                }
                (capture / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8")
                (capture / "W9_CAPTURE_COMPLETE").write_text(
                    "W9_CAPTURE_COMPLETE\n", encoding="utf-8")
            arms[arm_name] = {
                "binary_sha256": binary, "model_sha256": model,
                "tokenizer_sha256": tokenizer, "prompt_sha256": prompt_hash,
                "configuration_sha256": configuration, "context": 8193,
                "capture": arm_name == "on",
            }
        manifest = {
            "schema": "glm52-w9-real-capture-manifest-v1",
            "binary_sha256": binary, "model_sha256": model,
            "tokenizer_sha256": tokenizer, "prompt_sha256": prompt_hash,
            "configuration_sha256": configuration,
            "arm_order": ["off", "on"], "arms": arms,
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "record.json"
            for payload in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.strict_json(path)

    def _selected_fixture(self, root: pathlib.Path) -> None:
        counts = array.array("I", [2048] * (len(MODULE.LAYERS) * 128))
        (root / "selected-count.u32").write_bytes(counts.tobytes())
        row = array.array("I", [0] + [8193] * 2047).tobytes()
        with (root / "selected.u32").open("wb") as handle:
            for _ in counts:
                handle.write(row)

    def test_selected_contract_and_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self._selected_fixture(root)
            result = MODULE.validate_selected(root, 8193)
            self.assertEqual(result["rows"], 1024)

            counts_path = root / "selected-count.u32"
            original_counts = counts_path.read_bytes()
            for bad_count in (0, 1, 2047, 2049):
                changed = bytearray(original_counts)
                changed[:4] = bad_count.to_bytes(4, "little")
                counts_path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    MODULE.validate_selected(root, 8193)
            counts_path.write_bytes(original_counts)

            selected_path = root / "selected.u32"
            with selected_path.open("r+b") as handle:
                handle.write((1).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "noncausal"):
                MODULE.validate_selected(root, 8193)

    def test_f32_rejects_nonfinite_and_wrong_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "values.f32"
            values = array.array("f", [0.0, 1.0, 2.0, 3.0])
            path.write_bytes(values.tobytes())
            MODULE.validate_f32(path, 16)
            with self.assertRaisesRegex(ValueError, "size"):
                MODULE.validate_f32(path, 20)
            values[2] = float("nan")
            path.write_bytes(values.tobytes())
            with self.assertRaisesRegex(ValueError, "non-finite"):
                MODULE.validate_f32(path, 16)

    def test_stable_descriptor_rejects_same_size_inplace_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "artifact"
            path.write_bytes(b"a" * 8192)
            original_read = MODULE.os.read
            changed = False

            def read_then_mutate(descriptor, count):
                nonlocal changed
                chunk = original_read(descriptor, count)
                if chunk and not changed:
                    changed = True
                    path.write_bytes(b"b" * 8192)
                return chunk

            with mock.patch.object(MODULE.os, "read", side_effect=read_then_mutate):
                with self.assertRaisesRegex(ValueError, "changed while hashed"):
                    MODULE.scan_regular(path)

    def test_capture_rejects_missing_extra_and_metadata_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.assertNotEqual(MODULE.CAPTURE_NAMES, set())
            (root / "unexpected").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "inventory"):
                MODULE.validate_capture(root)

    def test_terminal_inventory_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            artifact = root / "artifact"
            artifact.write_bytes(b"original")
            rows = MODULE.inventory(root)
            MODULE.verify_inventory(root, rows)
            artifact.write_bytes(b"mutated!")
            with self.assertRaisesRegex(ValueError, "mutation"):
                MODULE.verify_inventory(root, rows)
            artifact.write_bytes(b"original")
            extra = root / "post-receipt"
            extra.write_bytes(b"added")
            with self.assertRaisesRegex(ValueError, "path set"):
                MODULE.verify_inventory(root, rows)

    def test_logit_publication_rejects_failure_and_substitution(self) -> None:
        path = pathlib.Path("/attempt/off/logits.sync0.start0.prompt8192.suffix8192")
        MODULE.validate_logit_publication(
            f"ds4: prefill logits dumped to {path}\n", path)
        with self.assertRaisesRegex(ValueError, "failure"):
            MODULE.validate_logit_publication(
                f"ds4: prefill logits dump failed for {path}: File exists\n", path)
        with self.assertRaisesRegex(ValueError, "binding"):
            MODULE.validate_logit_publication(
                "ds4: prefill logits dumped to /attempt/off/substitute\n", path)

    def test_complete_paired_score_and_cross_phase_rewrite(self) -> None:
        compact_sizes = {"kv.f32": 16, "query.f32": 16,
                         "selected.u32": 128 * 2048 * 4,
                         "selected-count.u32": 128 * 4}
        with mock.patch.object(MODULE, "LAYERS", (0,)), \
                mock.patch.object(MODULE, "LOGIT_COUNT", 4), \
                mock.patch.object(MODULE, "CAPTURE_SIZES", compact_sizes), \
                mock.patch.object(MODULE, "CAPTURE_NAMES",
                                  set(compact_sizes) |
                                  {"metadata.json", "W9_CAPTURE_COMPLETE"}):
            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                manifest = self._paired_fixture(root)
                rows, summary = MODULE.score(root, manifest)
                self.assertEqual(summary["verdict"], "PASS")
                self.assertEqual([row["arm"] for row in rows], ["off", "on"])

            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                manifest = self._paired_fixture(root)
                real_inventory = MODULE.inventory
                calls = 0

                def inventory_then_rewrite(path):
                    nonlocal calls
                    result = real_inventory(path)
                    calls += 1
                    if calls == 2:
                        target = root / "on/logits.sync0.start0.prompt8192.suffix8192"
                        target.write_bytes(array.array("f", [1.0, 0.0, 0.0, 0.0]).tobytes())
                    return result

                with mock.patch.object(MODULE, "inventory", side_effect=inventory_then_rewrite):
                    with self.assertRaisesRegex(ValueError, "arm/final inventory"):
                        MODULE.score(root, manifest)

    def test_scorer_contract_is_fixed(self) -> None:
        source = SCORER.read_text(encoding="utf-8")
        for required in (
            "final_logits_byte_identical",
            "matched_inputs_and_configuration",
            "on_capture_exact_and_finite",
            "selected_padding",
            "W9_CAPTURE_COMPLETE",
            "terminal-receipt.json",
            "OOM/Xid evidence",
        ):
            self.assertIn(required, source)

    def test_harness_runs_matched_contained_arms(self) -> None:
        source = HARNESS.read_text(encoding="utf-8")
        for required in (
            "glm_safe_run.sh",
            "glm_cgroup_run.sh",
            "DS4_GLM_W9_CAPTURE_DIR",
            "DS4_GLM_LOGIT_DUMP_ALL=1",
            "GLM_SAFE_MEMORY_HIGH_GIB=78",
            "GLM_SAFE_KILL_FLOOR_GIB=24",
            "GLM_SAFE_MIN_START_GIB=110",
            "randomness_receipt",
            "for arm in \"${arms[@]}\"",
            "--failure-reason",
            "--verify-terminal",
            '[[ $(sha "$MODEL") == "$MODEL_SHA256" ]]',
            "MODEL_IDENTITY",
            "prefill logits dump failed",
            "prefill logits dumped to",
        ):
            self.assertIn(required, source)
        self.assertIn('if [[ $arm == on ]]', source)
        self.assertNotIn("-n 0", source)

        match = re.search(r"(?ms)^arm_output_path\(\) \{.*?^\}", source)
        self.assertIsNotNone(match)
        script = "set -u\n" + match.group(0) + "\n" + \
            "arm_output_path off /tmp/root\narm_output_path on /tmp/root\n"
        result = subprocess.run(["bash", "-c", script], text=True,
                                capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "/tmp/root/off\n/tmp/root/on\n")

    def test_exact_cli_arguments_parse_without_loading_a_model(self) -> None:
        binary = pathlib.Path("/home/bmarti44/.cache/glm52-w9-9ebc0f2-runtime/ds4-server")
        if not binary.is_file():
            self.skipTest("frozen candidate binary unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            prompt = root / "prompt.txt"
            prompt.write_text("probe", encoding="utf-8")
            command = [str(binary), "--cuda", "-m", str(root / "missing.gguf"),
                       "--raw-prompt", "--prompt-file", str(prompt), "-c", "8193",
                       "--temp", "0", "--dump-logits", str(root / "next.json"),
                       "--ssd-streaming", "--ssd-streaming-cache-experts", "40GB"]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("invalid value", result.stderr)
            self.assertNotIn("unknown option", result.stderr)


if __name__ == "__main__":
    unittest.main()
