#!/usr/bin/env python3
"""Contracts for scripts/lib/switch_generic.py, the generic switch-alias path.

A new model alias needs only a profile carrying ``switch_alias`` plus a
``"switch": {"mode": "generic", ...}`` block. These tests pin the fail-closed
rules the bash switch relies on: legacy aliases can never be claimed
generically, discovery rejects duplicates and unknown keys, estimated profiles
are never servable, and the sampled/full digest walk rejects tampering.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import switch_generic as generic  # noqa: E402

MODULE = ROOT / "scripts/lib/switch_generic.py"
SPARK_HOST = ROOT / "configs/hosts/spark-aba1.json"
PRODUCTION_REPO = "/home/bmarti44/spark-deepseek-v4-flash"
PRODUCTION_STATE = "/home/dsv4/ds4-project/engine-switch"
LEGACY = ("dsv4", "glm52", "qwen38", "qwen38-1m", "laguna", "glm53-1m")


def generic_profile(alias: str = "newmodel", **switch: object) -> dict:
    block = {"mode": "generic", "served_model_id": "new-model",
             "health_path": None, "context_check": None}
    block.update(switch)
    return {"switch_alias": alias, "switch": block}


def production_plan(alias: str = "dsv41flash") -> dict:
    return generic.build_plan(
        alias, repo=PRODUCTION_REPO, state=PRODUCTION_STATE, port=8013,
        host_path=SPARK_HOST,
    )


class AliasRules(unittest.TestCase):
    def test_legacy_aliases_are_never_generic(self) -> None:
        for alias in LEGACY:
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(generic.SwitchError, "legacy"):
                    generic.check_alias(alias)
                with self.assertRaises(generic.SwitchError):
                    production_plan(alias)

    def test_reserved_verbs_and_malformed_aliases_are_rejected(self) -> None:
        for alias in ("status", "stop", "restore", "render", "", "Bad",
                      "-lead", "a" * 33, "../x", "x y", "x_y"):
            with self.subTest(alias=alias):
                with self.assertRaises(generic.SwitchError):
                    generic.check_alias(alias)

    def test_valid_alias_shapes_are_accepted(self) -> None:
        for alias in ("dsv41flash", "kimi-k3", "qwen3.8-max", "a" * 32):
            generic.check_alias(alias)


class Discovery(unittest.TestCase):
    def test_duplicate_alias_fails_closed(self) -> None:
        entries = [
            ("model-a", "cuda-spark-128g.json", generic_profile()),
            ("model-b", "cuda-spark-128g.json", generic_profile()),
        ]
        with self.assertRaisesRegex(generic.SwitchError, "claimed by"):
            generic.discover("newmodel", entries)

    def test_unknown_alias_fails_closed(self) -> None:
        entries = [("model-a", "x.json", generic_profile("other"))]
        with self.assertRaisesRegex(generic.SwitchError, "no profile"):
            generic.discover("newmodel", entries)

    def test_profile_without_generic_switch_block_is_rejected(self) -> None:
        profile = generic_profile()
        del profile["switch"]
        with self.assertRaisesRegex(generic.SwitchError, "generic"):
            generic.discover("newmodel", [("model-a", "x.json", profile)])

    def test_unique_generic_profile_is_found(self) -> None:
        entries = [
            ("model-a", "x.json", generic_profile("other")),
            ("model-b", "y.json", generic_profile()),
            ("model-c", "z.json", {"switch_alias": None}),
        ]
        relpath, profile = generic.discover("newmodel", entries)
        self.assertEqual(relpath, "configs/profiles/model-b/y.json")
        self.assertEqual(profile["switch"]["served_model_id"], "new-model")

    def test_committed_dsv41flash_is_the_only_generic_alias_claim(self) -> None:
        relpath, profile = generic.discover(
            "dsv41flash", generic.committed_entries())
        self.assertEqual(
            relpath,
            "configs/profiles/deepseek-v4.1-flash/cuda-spark-128g-1m.json")
        self.assertEqual(profile["switch"]["mode"], "generic")


class SwitchBlock(unittest.TestCase):
    def test_unknown_switch_key_fails_closed(self) -> None:
        block = generic_profile(surprise=1)["switch"]
        with self.assertRaisesRegex(generic.SwitchError, "unknown"):
            generic.validate_switch_block(block)

    def test_missing_mode_or_served_id_fails_closed(self) -> None:
        for broken in ({"served_model_id": "x"}, {"mode": "generic"},
                       {"mode": "legacy", "served_model_id": "x"},
                       {"mode": "generic", "served_model_id": ""},
                       {"mode": "generic", "served_model_id": "a b"}):
            with self.subTest(block=broken):
                with self.assertRaises(generic.SwitchError):
                    generic.validate_switch_block(broken)

    def test_health_path_must_be_a_local_path(self) -> None:
        for bad in ("health", "http://x/health", "/a b", "/../x", 5):
            with self.subTest(path=bad):
                with self.assertRaises(generic.SwitchError):
                    generic.validate_switch_block(
                        generic_profile(health_path=bad)["switch"])
        block = generic.validate_switch_block(
            generic_profile(health_path="/health")["switch"])
        self.assertEqual(block["health_path"], "/health")

    def test_context_check_kinds_are_closed(self) -> None:
        good = (
            {"kind": "slots", "slots": 4, "n_ctx": 262144},
            {"kind": "model_card", "max_model_len": 262144},
        )
        for spec in good:
            generic.validate_switch_block(
                generic_profile(context_check=spec)["switch"])
        for spec in ({"kind": "slots", "slots": 4},
                     {"kind": "slots", "slots": 0, "n_ctx": 1},
                     {"kind": "model_card", "max_model_len": "1"},
                     {"kind": "mystery"},
                     {"kind": "slots", "slots": 1, "n_ctx": 1, "extra": 1},
                     "slots"):
            with self.subTest(spec=spec):
                with self.assertRaises(generic.SwitchError):
                    generic.validate_switch_block(
                        generic_profile(context_check=spec)["switch"])


class ProductionPlan(unittest.TestCase):
    def test_dsv41flash_plan_carries_the_profile_contract(self) -> None:
        plan = production_plan()
        self.assertEqual(plan["alias"], "dsv41flash")
        self.assertEqual(plan["status"], "estimated")
        self.assertEqual(
            plan["relpath"],
            "configs/profiles/deepseek-v4.1-flash/cuda-spark-128g-1m.json")
        self.assertEqual(
            plan["binary"], "/home/bmarti44/.cache/ds4-v41-0aaea5a2/ds4-server")
        self.assertEqual(
            plan["model"],
            "/home/bmarti44/models/deepseek-v4.1-flash/"
            "DeepSeek-V4.1-Flash-Q2.gguf")
        self.assertEqual(plan["unit"], "dsv41-engine.service")
        self.assertEqual(plan["unit_name"], "dsv41-engine")
        self.assertEqual(plan["user"], "bmarti44")
        self.assertEqual(plan["log_name"], "dsv41")
        self.assertEqual(plan["required_gib"], "110")
        self.assertEqual(plan["floor_gib"], "10")
        self.assertEqual(plan["timeout"], "1800")
        self.assertEqual(plan["served_model_id"], "deepseek-v4.1-flash")
        self.assertEqual(plan["health_path"], "")
        self.assertEqual(plan["context_check"], "")
        self.assertEqual(plan["context_path"], "")
        self.assertEqual(
            plan["binary_sha256"],
            json.loads((ROOT / "configs/build-manifests/ds4-v41.json")
                       .read_text())["binaries"]["ds4-server"]["sha256"])

    def test_plan_rejects_a_host_that_disagrees_with_the_switch(self) -> None:
        with self.assertRaisesRegex(generic.SwitchError, "state"):
            generic.build_plan(
                "dsv41flash", repo=PRODUCTION_REPO, state="/tmp/elsewhere",
                port=8013, host_path=SPARK_HOST)
        with self.assertRaisesRegex(generic.SwitchError, "port"):
            generic.build_plan(
                "dsv41flash", repo=PRODUCTION_REPO, state=PRODUCTION_STATE,
                port=8014, host_path=SPARK_HOST)

    def test_test_paths_override_host_roots(self) -> None:
        plan = generic.build_plan(
            "dsv41flash", repo=str(ROOT), state="/tmp/t", port=8013,
            host_path=SPARK_HOST,
            test_paths={"cache_root": "/tmp/t/cache", "model_root": "/tmp/t/models"})
        self.assertEqual(plan["binary"], "/tmp/t/cache/ds4-v41-0aaea5a2/ds4-server")
        self.assertTrue(plan["model"].startswith("/tmp/t/models/"))

    def test_plan_cli_emits_nul_delimited_pairs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", str(MODULE), "--repo", PRODUCTION_REPO,
             "--state", PRODUCTION_STATE, "--port", "8013",
             "--host", str(SPARK_HOST), "plan", "dsv41flash"],
            capture_output=True, check=False, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        pairs = dict(item.split(b"=", 1) for item in
                     result.stdout.split(b"\0") if item)
        self.assertEqual(pairs[b"alias"], b"dsv41flash")
        self.assertEqual(pairs[b"served_model_id"], b"deepseek-v4.1-flash")
        self.assertNotIn(b"digest_checks", pairs)

    def test_plan_cli_fails_closed_on_legacy_alias(self) -> None:
        result = subprocess.run(
            [sys.executable, "-B", str(MODULE), "--repo", PRODUCTION_REPO,
             "--state", PRODUCTION_STATE, "--port", "8013",
             "--host", str(SPARK_HOST), "plan", "glm53-1m"],
            capture_output=True, text=True, check=False, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("legacy", result.stderr)

    def test_unsupported_launch_placeholders_are_rejected(self) -> None:
        profile = {"launch": {"args": ["--x", "{model_root}"], "env": {}}}
        with self.assertRaisesRegex(generic.SwitchError, "placeholder"):
            generic.check_launch_placeholders(profile)
        profile = {"launch": {"args": ["-m", "{model}", "--port", "{port}"],
                              "env": {"HOME": "{model}"}}}
        with self.assertRaisesRegex(generic.SwitchError, "placeholder"):
            generic.check_launch_placeholders(profile)
        generic.check_launch_placeholders(
            {"launch": {"args": ["-m", "{model}", "--port", "{port}"],
                        "env": {"PYTHONPATH": "{repo}/scripts/lib"}}})


class ArtifactVerification(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.binary = self.tmp / "server"
        self.model = self.tmp / "model.gguf"
        self.binary.write_bytes(b"fixture-binary\n")
        self.model.write_bytes(b"M" * 4096 + b"tail")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def plan(self, **overrides: object) -> dict:
        info = os.stat(self.model)
        plan = {
            "alias": "newmodel",
            "status": "qualified",
            "binary": str(self.binary),
            "model": str(self.model),
            "mmproj": "",
            "draft_model": "",
            "digest_checks": [
                {"path": str(self.binary),
                 "sha256": hashlib.sha256(self.binary.read_bytes()).hexdigest()},
                {"path": str(self.model), "identity": {
                    "first_bytes": 1024,
                    "first_bytes_sha256": hashlib.sha256(
                        self.model.read_bytes()[:1024]).hexdigest(),
                    "size_bytes": info.st_size,
                    "device": info.st_dev,
                    "inode": info.st_ino,
                }},
            ],
        }
        plan.update(overrides)
        return plan

    def test_estimated_profile_is_refused_before_any_file_is_read(self) -> None:
        plan = self.plan(status="estimated")
        plan["digest_checks"] = [{"path": "/nonexistent/x", "sha256": "0" * 64}]
        with self.assertRaisesRegex(generic.SwitchError, "estimated"):
            generic.verify_artifacts(plan)
        with self.assertRaisesRegex(generic.SwitchError, "unsupported"):
            generic.verify_artifacts(self.plan(status="unsupported"))

    def test_committed_dsv41flash_is_refused_while_estimated(self) -> None:
        with self.assertRaisesRegex(generic.SwitchError, "estimated"):
            generic.verify_artifacts(production_plan())

    def test_valid_full_and_sampled_digests_pass(self) -> None:
        identities = generic.verify_artifacts(self.plan())
        self.assertEqual(sorted(identities), sorted([str(self.binary), str(self.model)]))
        info = os.lstat(self.model)
        self.assertEqual(identities[str(self.model)][:2], [info.st_dev, info.st_ino])

    def test_tampered_binary_is_rejected(self) -> None:
        plan = self.plan()
        self.binary.write_bytes(b"fixture-binarY\n")
        with self.assertRaisesRegex(generic.SwitchError, "hash is not approved"):
            generic.verify_artifacts(plan)

    def test_tampered_model_prefix_is_rejected(self) -> None:
        plan = self.plan()
        with open(self.model, "r+b") as stream:
            stream.write(b"X")
        with self.assertRaisesRegex(generic.SwitchError, "prefix hash"):
            generic.verify_artifacts(plan)

    def test_replaced_model_inode_is_rejected(self) -> None:
        plan = self.plan()
        payload = self.model.read_bytes()
        self.model.unlink()
        spare = self.tmp / "spare"
        spare.write_bytes(b"")  # keep the old inode number from being reused
        self.model.write_bytes(payload)
        with self.assertRaisesRegex(generic.SwitchError, "stat identity"):
            generic.verify_artifacts(plan)

    def test_symlinked_artifact_is_rejected(self) -> None:
        plan = self.plan()
        real = self.tmp / "real-server"
        self.binary.rename(real)
        self.binary.symlink_to(real)
        with self.assertRaises(generic.SwitchError):
            generic.verify_artifacts(plan)

    def test_digest_target_outside_the_launch_artifacts_is_rejected(self) -> None:
        plan = self.plan()
        stray = self.tmp / "elsewhere" / "x"
        stray.parent.mkdir()
        stray.write_bytes(b"x")
        plan["digest_checks"].append({
            "path": str(stray),
            "sha256": hashlib.sha256(b"x").hexdigest()})
        with self.assertRaisesRegex(generic.SwitchError, "escapes"):
            generic.verify_artifacts(plan)

    def test_duplicate_or_missing_pins_are_rejected(self) -> None:
        plan = self.plan()
        plan["digest_checks"].append(copy.deepcopy(plan["digest_checks"][0]))
        with self.assertRaisesRegex(generic.SwitchError, "repeats"):
            generic.verify_artifacts(plan)
        plan = self.plan()
        plan["digest_checks"] = plan["digest_checks"][:1]
        with self.assertRaisesRegex(generic.SwitchError, "model"):
            generic.verify_artifacts(plan)
        plan = self.plan()
        plan["digest_checks"] = plan["digest_checks"][1:]
        with self.assertRaisesRegex(generic.SwitchError, "binary"):
            generic.verify_artifacts(plan)

    def test_malformed_identity_is_rejected(self) -> None:
        plan = self.plan()
        plan["digest_checks"][1]["identity"]["first_bytes"] = 0
        with self.assertRaises(generic.SwitchError):
            generic.verify_artifacts(plan)
        plan = self.plan()
        del plan["digest_checks"][1]["identity"]["inode"]
        with self.assertRaises(generic.SwitchError):
            generic.verify_artifacts(plan)

    def test_revalidate_detects_a_change_after_approval(self) -> None:
        plan = self.plan()
        identities = generic.verify_artifacts(plan)
        generic.revalidate(plan, identities)
        os.utime(self.binary, ns=(1, 1))
        with self.assertRaisesRegex(generic.SwitchError, "changed after"):
            generic.revalidate(plan, identities)

    def test_revalidate_requires_binary_and_model(self) -> None:
        plan = self.plan()
        identities = generic.verify_artifacts(plan)
        identities.pop(str(self.model))
        with self.assertRaises(generic.SwitchError):
            generic.revalidate(plan, identities)


class ServingChecks(unittest.TestCase):
    DSV4 = json.dumps({"data": [{"id": "deepseek-v4-flash"}]})
    DSV41 = json.dumps({"data": [{"id": "deepseek-v4.1-flash"}]})

    def test_models_check_requires_the_exact_served_id(self) -> None:
        generic.check_models("deepseek-v4.1-flash", self.DSV41)
        with self.assertRaises(generic.SwitchError):
            generic.check_models("deepseek-v4.1-flash", self.DSV4)
        with self.assertRaises(generic.SwitchError):
            generic.check_models("deepseek-v4.1-flash", "not json")

    def test_slots_context_check(self) -> None:
        spec = json.dumps({"kind": "slots", "slots": 2, "n_ctx": 100})
        generic.check_context(spec, json.dumps([{"n_ctx": 100}, {"n_ctx": 100}]))
        with self.assertRaises(generic.SwitchError):
            generic.check_context(spec, json.dumps([{"n_ctx": 100}]))
        with self.assertRaises(generic.SwitchError):
            generic.check_context(spec, json.dumps([{"n_ctx": 100}, {"n_ctx": 99}]))

    def test_model_card_context_check(self) -> None:
        spec = json.dumps({"kind": "model_card", "max_model_len": 262144,
                           "served_model_id": "glm-5.3-flash"})
        good = json.dumps({"data": [{"id": "glm-5.3-flash", "max_model_len": 262144}]})
        generic.check_context(spec, good)
        bad = json.dumps({"data": [{"id": "glm-5.3-flash", "max_model_len": 4096}]})
        with self.assertRaises(generic.SwitchError):
            generic.check_context(spec, bad)


if __name__ == "__main__":
    unittest.main()
