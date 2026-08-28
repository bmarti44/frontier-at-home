#!/usr/bin/env python3
"""Spending the holdout budget must be keyed on the ROWS, not on the config.

`select_indices` draws holdout rows with `random.Random(SEED)` where SEED is the
module constant 42. The draw is therefore identical on every run: GSM8K holdout is
always the same 100 shuffled indices, MMLU-Pro holdout always the same stratified
selection. The ledger, however, keys on
`(namespace, stack_label, suite, config_digest)` and refuses only an exact repeat
of that tuple.

So any change that moves the config digest mints a ledger-valid "new" holdout entry
over examples that have already been seen. That is not hypothetical bookkeeping: the
0731 work alone added three new digest inputs (thinking_mode, max_tokens,
request_timeout_s), each of which produces a fresh digest over the identical rows.
Repeatedly measuring the same holdout under tweaked configs and keeping the best is
exactly the overfitting the ledger exists to prevent, and the ledger currently
records it as clean.

The fix asserted here is to record what was actually spent -- a digest over the
selected row identities -- so a reader and an auditor can see that two "different"
entries consumed the same examples.
"""

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "31_bench_accuracy.py"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_rowset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class HoldoutRowsetTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_holdout_draw_is_deterministic_across_runs(self):
        # Establishes the premise: nothing about the config changes the rows.
        rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(1319)]
        first = self.module.select_indices("gsm8k", "holdout", rows)
        second = self.module.select_indices("gsm8k", "holdout", rows)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        dev = self.module.select_indices("gsm8k", "dev", rows)
        self.assertFalse(set(first) & set(dev), "dev and holdout must be disjoint")

    def test_rowset_digest_is_exposed(self):
        rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(1319)]
        indices = self.module.select_indices("gsm8k", "holdout", rows)
        digest = self.module.holdout_rowset_digest("gsm8k", "holdout", rows, indices)
        self.assertRegex(digest, r"\A[0-9a-f]{64}\Z")

    def test_rowset_digest_depends_only_on_the_rows_drawn(self):
        rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(1319)]
        indices = self.module.select_indices("gsm8k", "holdout", rows)
        a = self.module.holdout_rowset_digest("gsm8k", "holdout", rows, indices)
        b = self.module.holdout_rowset_digest("gsm8k", "holdout", rows, list(indices))
        self.assertEqual(a, b, "the digest must not depend on list identity")

        mutated = list(indices)
        mutated[0] = (mutated[0] + 1) % len(rows)
        c = self.module.holdout_rowset_digest("gsm8k", "holdout", rows, mutated)
        self.assertNotEqual(a, c, "a different draw must produce a different digest")

    def test_profile_id_extends_the_digest_only_when_given(self):
        """--profile-id is a measurement identity (docs/PROFILE-SCHEMA.md).

        Profile-less invocations must keep producing payloads with the exact
        pre-profile key set so frozen digests stay reproducible in schema;
        a named profile adds the key and therefore mints a new digest.
        """
        import argparse
        module = self.module
        base_kwargs = dict(
            config_evidence=None,
            stack_label="stack",
            suite="gsm8k",
            split="holdout",
            extra_body=None,
            max_tokens=16384,
            thinking_mode="chat",
            encoder="deepseek",
            reasoning_effort=None,
            request_timeout=600,
            profile_id=None,
        )
        digest, payload, _ = module.derive_config_digest(
            argparse.Namespace(**base_kwargs)
        )
        self.assertIsNone(digest)
        self.assertIsNone(payload)
        # With config evidence absent we cannot build a full payload here;
        # assert the payload key contract at the source level instead.
        source = SCRIPT.read_text()
        self.assertIn('if args.profile_id is not None:', source)
        self.assertIn('digest_payload["profile_id"] = args.profile_id', source)

    def test_ledger_entry_carries_the_rowset_digest(self):
        # The ledger row is what an auditor reads; the digest has to be in it.
        module = self.module
        written = {}

        def fake_write_json(path, value):
            written["value"] = value

        module.write_json = fake_write_json
        module.load_holdout_entries = lambda: []

        class _Stream:
            def close(self):
                pass

        module.acquire_holdout_ledger = lambda: _Stream()
        module.append_holdout_ledger(
            {
                "ledger_namespace": "ns",
                "stack_label": "s",
                "suite": "gsm8k",
                "config_digest": "a" * 64,
                "holdout_rowset_sha256": "b" * 64,
                "phase": "started",
            },
            refuse_existing=True,
        )
        self.assertEqual(
            written["value"][0]["holdout_rowset_sha256"], "b" * 64
        )

    def test_ledger_refuses_the_same_rows_under_a_new_config_digest(self):
        module = self.module
        existing = [
            {
                "ledger_namespace": "ns",
                "stack_label": "s",
                "suite": "gsm8k",
                "config_digest": "a" * 64,
                "holdout_rowset_sha256": "b" * 64,
                "phase": "completed",
            }
        ]
        module.load_holdout_entries = lambda: list(existing)
        module.write_json = lambda path, value: None

        class _Stream:
            def close(self):
                pass

        module.acquire_holdout_ledger = lambda: _Stream()

        with self.assertRaises(module.HoldoutAlreadyRun) as caught:
            module.append_holdout_ledger(
                {
                    "ledger_namespace": "ns",
                    "stack_label": "s",
                    "suite": "gsm8k",
                    # A different config digest -- previously enough to pass.
                    "config_digest": "c" * 64,
                    # But the SAME rows.
                    "holdout_rowset_sha256": "b" * 64,
                    "phase": "started",
                },
                refuse_existing=True,
            )
        self.assertIn("rows", str(caught.exception).lower())



class AuditorRowsetTests(unittest.TestCase):
    """The auditor must reject a repeat look, not just a malformed receipt."""

    def load_auditor(self, ledger_entries, tmpdir):
        ledger = Path(tmpdir) / "holdout-ledger.json"
        ledger.write_text(json.dumps(ledger_entries), encoding="utf-8")
        spec = importlib.util.spec_from_file_location(
            "audit_accuracy", ROOT / "scripts" / "36_audit_accuracy.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.LEDGER = ledger
        return module

    def _entry(self, digest, rowset, phase):
        return {
            "ledger_namespace": "ns",
            "stack_label": "s",
            "suite": "gsm8k",
            "config_digest": digest,
            "holdout_rowset_sha256": rowset,
            "phase": phase,
            "started_at": "2026-08-10T00:00:00+00:00",
            "completed_at": "2026-08-10T01:00:00+00:00",
        }

    def test_rejects_same_rows_under_a_new_config_digest(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            entries = [
                self._entry("a" * 64, "1" * 64, "started"),
                self._entry("a" * 64, "1" * 64, "completed"),
                self._entry("c" * 64, "1" * 64, "started"),
                self._entry("c" * 64, "1" * 64, "completed"),
            ]
            module = self.load_auditor(entries, tmp)
            document = {
                "ledger_namespace": "ns",
                "stack_label": "s",
                "suite": "gsm8k",
                "config_digest": "c" * 64,
                "rowset_sha256": "1" * 64,
            }
            problems = module.audit_holdout_ledger(Path("acc-x.json"), document)
            self.assertTrue(
                any("already" in p and "spent" in p for p in problems),
                f"expected a repeat-look rejection, got {problems!r}",
            )

    def test_accepts_a_first_look(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            entries = [
                self._entry("a" * 64, "1" * 64, "started"),
                self._entry("a" * 64, "1" * 64, "completed"),
            ]
            module = self.load_auditor(entries, tmp)
            document = {
                "ledger_namespace": "ns",
                "stack_label": "s",
                "suite": "gsm8k",
                "config_digest": "a" * 64,
                "rowset_sha256": "1" * 64,
            }
            problems = module.audit_holdout_ledger(Path("acc-x.json"), document)
            self.assertEqual(problems, [], f"unexpected problems: {problems!r}")

    def test_rejects_a_result_whose_rowset_contradicts_its_ledger_entry(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            entries = [
                self._entry("a" * 64, "1" * 64, "started"),
                self._entry("a" * 64, "1" * 64, "completed"),
            ]
            module = self.load_auditor(entries, tmp)
            document = {
                "ledger_namespace": "ns",
                "stack_label": "s",
                "suite": "gsm8k",
                "config_digest": "a" * 64,
                "rowset_sha256": "2" * 64,
            }
            problems = module.audit_holdout_ledger(Path("acc-x.json"), document)
            self.assertTrue(
                any("records rowset" in p for p in problems),
                f"expected a rowset contradiction, got {problems!r}",
            )

    def test_a_different_stack_on_the_same_rows_is_legal(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            other = self._entry("a" * 64, "1" * 64, "completed")
            other["stack_label"] = "other-stack"
            entries = [
                other,
                self._entry("c" * 64, "1" * 64, "started"),
                self._entry("c" * 64, "1" * 64, "completed"),
            ]
            module = self.load_auditor(entries, tmp)
            document = {
                "ledger_namespace": "ns",
                "stack_label": "s",
                "suite": "gsm8k",
                "config_digest": "c" * 64,
                "rowset_sha256": "1" * 64,
            }
            problems = module.audit_holdout_ledger(Path("acc-x.json"), document)
            self.assertEqual(
                problems,
                [],
                "comparing a different stack on the same holdout is the intended "
                f"use and must not be rejected; got {problems!r}",
            )



class DecisionOutputIsolationTests(unittest.TestCase):
    """A new candidate must not overwrite the frozen terminal decision."""

    def test_decision_id_is_validated_and_redirects_output(self):
        source = (ROOT / "scripts" / "34_decision.py").read_text(encoding="utf-8")
        self.assertIn("--decision-id", source)
        self.assertIn('f"decision{suffix}.json"', source)
        self.assertIn('f"DECISION{suffix.upper()}.md"', source)

    def test_spec_names_its_output_and_forbids_the_frozen_paths(self):
        spec = json.loads(
            (ROOT / "configs" / "decision-specs" / "dsv4-0731.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(spec["decision_id"], "0731")
        self.assertEqual(
            spec["output_destination"]["machine"], "results/decision-0731.json"
        )
        self.assertIn(
            "results/decision.json", spec["output_destination"]["must_not_write"]
        )

    def test_spec_states_that_it_is_not_yet_executable(self):
        # Guards against this preregistration being read as a working config.
        spec = json.loads(
            (ROOT / "configs" / "decision-specs" / "dsv4-0731.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(spec["status"], "SPECIFICATION_ONLY_NOT_YET_EXECUTABLE")
        self.assertIn("what_is_NOT_implemented", spec["honest_status"])

    def test_spec_preregisters_gates_matching_the_measured_baseline_rates(self):
        spec = json.loads(
            (ROOT / "configs" / "decision-specs" / "dsv4-0731.v1.json").read_text(
                encoding="utf-8"
            )
        )
        gates = spec["preregistered_validity_gates"]
        self.assertEqual(gates["truncated_fraction_max"]["mmlu-pro"], 0.055)
        self.assertEqual(gates["truncated_fraction_max"]["gsm8k"], 0.010)
        self.assertEqual(gates["request_failures_max"], 0)


if __name__ == "__main__":
    unittest.main()
