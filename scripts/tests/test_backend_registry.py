#!/usr/bin/env python3
"""Contracts for architecture-neutral serving backend dispatch."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs/backends.json"
CATALOG = ROOT / "models/catalog.json"
DISPATCHER = ROOT / "scripts/91_serve.sh"


class BackendRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        cls.backends = cls.registry["backends"]

    def run_dispatcher(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(DISPATCHER), *arguments],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_registry_parses_and_has_current_schema(self) -> None:
        self.assertEqual(self.registry["schema_version"], 1)
        self.assertEqual(set(self.registry), {"schema_version", "backends"})
        self.assertIsInstance(self.backends, dict)

    def test_backend_keys_exactly_match_catalog_claim_backends(self) -> None:
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(set(self.backends), set(catalog["claim_backends"]))

    def test_registered_serve_scripts_exist_are_tracked_and_executable(self) -> None:
        tracked = set(
            subprocess.run(
                ["git", "-C", str(ROOT), "ls-files"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.splitlines()
        )
        for backend_name, backend in self.backends.items():
            self.assertIsInstance(backend["implemented"], bool)
            self.assertIsInstance(backend["platform"], str)
            self.assertTrue(backend["platform"])
            self.assertIsInstance(backend["serve_scripts"], dict)
            for model, relative in backend["serve_scripts"].items():
                with self.subTest(backend=backend_name, model=model):
                    self.assertFalse(Path(relative).is_absolute())
                    self.assertIn(relative, tracked)
                    self.assertTrue((ROOT / relative).is_file())
                    self.assertTrue(os.access(ROOT / relative, os.X_OK))

    def test_cuda_maps_the_known_models(self) -> None:
        self.assertTrue(self.backends["cuda"]["implemented"])
        self.assertEqual(
            self.backends["cuda"]["serve_scripts"],
            {
                "qwen38": "scripts/22_serve_qwen38.sh",
                "qwen38-sglang": "scripts/23_serve_qwen38_sglang.sh",
                "laguna-s-2.1": "scripts/25_serve_laguna.sh",
            },
        )

    def test_unimplemented_backends_have_no_serve_scripts(self) -> None:
        for name, backend in self.backends.items():
            if not backend["implemented"]:
                with self.subTest(backend=name):
                    self.assertEqual(backend["serve_scripts"], {})

    def test_dispatcher_prints_registered_command_without_executing(self) -> None:
        result = self.run_dispatcher(
            "--model", "qwen38", "--backend", "cuda",
            "--print-command", "status",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(), "scripts/22_serve_qwen38.sh status"
        )

    def test_dispatcher_rejects_unknown_backend_with_valid_names(self) -> None:
        result = self.run_dispatcher(
            "--model", "qwen38", "--backend", "unknown",
            "--print-command", "status",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("valid backends:", result.stderr)
        for name in self.backends:
            self.assertIn(name, result.stderr)

    def test_dispatcher_rejects_unimplemented_backend(self) -> None:
        result = self.run_dispatcher(
            "--model", "qwen38", "--backend", "rocm",
            "--print-command", "status",
        )
        self.assertEqual(result.returncode, 3)
        self.assertIn(
            "backend rocm is registered but not implemented on this host; "
            "see docs/BACKEND-CONTRACT.md",
            result.stderr,
        )

    def test_dispatcher_rejects_model_missing_from_backend(self) -> None:
        result = self.run_dispatcher(
            "--model", "not-registered", "--backend", "cuda",
            "--print-command", "status",
        )
        self.assertEqual(result.returncode, 4)
        self.assertIn("configs/backends.json", result.stderr)
        self.assertIn("scripts/90_scaffold_model.sh", result.stderr)


if __name__ == "__main__":
    unittest.main()
