#!/usr/bin/env python3
"""Regression tests for atomic production lock provisioning."""

from __future__ import annotations

import fcntl
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PROVISIONER = ROOT / "scripts/68_provision_runtime_locks.py"


def load_provisioner():
    spec = importlib.util.spec_from_file_location("runtime_lock_provisioner", PROVISIONER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeLockProvisionerTests(unittest.TestCase):
    def test_legacy_holder_blocks_before_current_lock_is_published(self):
        provisioner = load_provisioner()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runtime = base / "runtime"
            legacy = runtime / "inference.lock"
            current_root = base / "current"
            current = current_root / "inference.lock"
            runtime.mkdir()
            legacy.touch()
            with legacy.open("a+b") as held:
                fcntl.flock(held, fcntl.LOCK_EX)
                with (
                    mock.patch.object(provisioner, "RUNTIME", runtime),
                    mock.patch.object(provisioner, "LEGACY", legacy),
                    mock.patch.object(provisioner, "CURRENT_ROOT", current_root),
                    mock.patch.object(provisioner, "CURRENT", current),
                    mock.patch.object(provisioner.os, "chown"),
                    mock.patch.object(provisioner.os, "fchown"),
                    mock.patch.object(provisioner, "validate_directory_identity"),
                    mock.patch.object(provisioner.os, "geteuid", return_value=0),
                    mock.patch.object(provisioner.sys, "argv", ["provisioner"]),
                    self.assertRaisesRegex(RuntimeError, "occupied"),
                ):
                    provisioner.main()
            self.assertFalse(current.exists())

    def test_success_publishes_both_locks_only_while_bridge_is_held(self):
        provisioner = load_provisioner()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runtime = base / "runtime"
            legacy = runtime / "inference.lock"
            current_root = base / "current"
            current = current_root / "inference.lock"
            with (
                mock.patch.object(provisioner, "RUNTIME", runtime),
                mock.patch.object(provisioner, "LEGACY", legacy),
                mock.patch.object(provisioner, "CURRENT_ROOT", current_root),
                mock.patch.object(provisioner, "CURRENT", current),
                mock.patch.object(provisioner.os, "chown"),
                mock.patch.object(provisioner.os, "fchown"),
                mock.patch.object(provisioner, "validate_visible_identity"),
                mock.patch.object(provisioner, "validate_directory_identity"),
                mock.patch.object(provisioner.os, "geteuid", return_value=0),
                mock.patch.object(provisioner.sys, "argv", ["provisioner"]),
            ):
                self.assertEqual(provisioner.main(), 0)
            for path in (legacy, current):
                self.assertTrue(path.is_file())
                with path.open("a+b") as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_preplanted_current_parent_symlink_is_not_followed(self):
        provisioner = load_provisioner()
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            runtime = base / "runtime"
            legacy = runtime / "inference.lock"
            target = base / "target"
            target.mkdir()
            original_mode = target.stat().st_mode
            current_root = base / "current"
            current_root.symlink_to(target, target_is_directory=True)
            current = current_root / "inference.lock"
            with (
                mock.patch.object(provisioner, "RUNTIME", runtime),
                mock.patch.object(provisioner, "LEGACY", legacy),
                mock.patch.object(provisioner, "CURRENT_ROOT", current_root),
                mock.patch.object(provisioner, "CURRENT", current),
                mock.patch.object(provisioner.os, "chown"),
                mock.patch.object(provisioner.os, "fchown"),
                mock.patch.object(provisioner, "validate_visible_identity"),
                mock.patch.object(provisioner, "validate_directory_identity"),
                mock.patch.object(provisioner.os, "geteuid", return_value=0),
                mock.patch.object(provisioner.sys, "argv", ["provisioner"]),
                self.assertRaises(OSError),
            ):
                provisioner.main()
            self.assertEqual(target.stat().st_mode, original_mode)
            self.assertEqual(list(target.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
