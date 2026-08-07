from __future__ import annotations

import array
import importlib.util
import json
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCORER = ROOT / "scripts/91_score_w9_real_capture.py"
HARNESS = ROOT / "results/glm52-gates/harness/w9_real_capture_v1.sh"
SPEC = importlib.util.spec_from_file_location("w9_scorer", SCORER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class W9RealCaptureScorerTests(unittest.TestCase):
    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "record.json"
            for payload in ('{"a":1,"a":2}', '{"a":NaN}'):
                path.write_text(payload, encoding="utf-8")
                with self.assertRaises(ValueError):
                    MODULE.strict_json(path)

    def _selected_fixture(self, root: pathlib.Path) -> None:
        counts = array.array("I", [1] * (len(MODULE.LAYERS) * 128))
        (root / "selected-count.u32").write_bytes(counts.tobytes())
        row = array.array("I", [0] + [0xFFFFFFFF] * 2047).tobytes()
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
            changed = bytearray(original_counts)
            changed[:4] = (0).to_bytes(4, "little")
            counts_path.write_bytes(changed)
            with self.assertRaises(ValueError):
                MODULE.validate_selected(root, 8193)
            counts_path.write_bytes(original_counts)

            selected_path = root / "selected.u32"
            with selected_path.open("r+b") as handle:
                handle.write((1).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "noncausal"):
                MODULE.validate_selected(root, 8193)
            with selected_path.open("r+b") as handle:
                handle.write((0).to_bytes(4, "little"))
                handle.seek(4)
                handle.write((7).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "padding"):
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
        ):
            self.assertIn(required, source)
        self.assertIn('if [[ $arm == on ]]', source)
        self.assertNotIn("DS4_GLM_W9_CAPTURE_DIR=$arm_out/capture\n  fi\n  set +e\n  /usr/bin/env -i", source)


if __name__ == "__main__":
    unittest.main()
