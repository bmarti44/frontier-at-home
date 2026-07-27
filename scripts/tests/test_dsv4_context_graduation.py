#!/usr/bin/env python3
"""Contracts for guarded DeepSeek context graduation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "scripts" / "57_dsv4_context_probe.py"
WORKER = ROOT / "scripts" / "58_dsv4_context_worker.sh"
SCHEDULER = ROOT / "scripts" / "59_schedule_dsv4_context.sh"


def load_probe():
    spec = importlib.util.spec_from_file_location("dsv4_context_probe", PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load context probe")
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


class ContextWorkerContractTests(unittest.TestCase):
    def test_worker_graduates_exact_rungs_and_restores_safe_profile(self):
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


if __name__ == "__main__":
    unittest.main()
