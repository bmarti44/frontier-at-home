"""Permission admission uses effective access checks, before model imports."""
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


class ServiceAccessContract(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("glm53_service_access")
        self.credentials = {"Uid": "995 995 995 995", "Gid": "982 982 982 982", "Groups": "982",
                            "CapInh": "0", "CapPrm": "0", "CapEff": "0", "CapBnd": "0", "CapAmb": "0",
                            "NoNewPrivs": "1"}

    def test_credential_gate_rejects_privilege_or_group_drift(self):
        self.api.validate_credentials(self.credentials, 995, 982)
        for key, value in (("Uid", "995 0 995 995"), ("Gid", "982 982 0 982"),
                           ("Groups", "982 1000"), ("CapInh", "1"), ("CapPrm", "1"),
                           ("CapEff", "1"), ("CapBnd", "1"), ("CapAmb", "1"), ("NoNewPrivs", "0")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.api.validate_credentials(dict(self.credentials, **{key: value}), 995, 982)
        with self.assertRaises(ValueError):
            self.api.validate_credentials({}, 995, 982)

    def test_production_entry_rejects_current_builder_before_inventory_access(self):
        with (mock.patch.object(self.api, "read_credentials", return_value=dict(self.credentials, Uid="1000 1000 1000 1000")),
              mock.patch.object(self.api, "verify_inventory", side_effect=AssertionError("inventory entered"))):
            with self.assertRaisesRegex(ValueError, "Uid|uid|identity"):
                self.api.verify_service_tree(Path("/nonexistent"), Path("/absent"), "a" * 64, 995, 982)

    def test_access_gate_checks_effective_acl_access_and_all_ancestors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            artifact = root / "engine.so"
            artifact.write_bytes(b"synthetic")
            def access(path, mode, *, effective_ids):
                self.assertTrue(effective_ids)
                return mode != os.W_OK
            with mock.patch.object(self.api.os, "access", side_effect=access) as checks:
                self.api.check_tree_access(root, 995)
                self.assertTrue(any(Path(call.args[0]) == root.parent and call.args[1] == os.W_OK for call in checks.call_args_list))
            for writable in (root.parent, root, artifact):
                with self.subTest(writable=writable):
                    with mock.patch.object(self.api.os, "access", side_effect=lambda p, mode, **kw: True if Path(p) == writable else mode != os.W_OK):
                        with self.assertRaisesRegex(ValueError, "writ"):
                            self.api.check_tree_access(root, 995)
            with mock.patch.object(self.api.os, "access", return_value=False):
                with self.assertRaisesRegex(ValueError, "read|search|access"):
                    self.api.check_tree_access(root, 995)

    def test_service_owned_readonly_files_and_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            with mock.patch.object(self.api.os, "access", side_effect=lambda p, mode, **kw: mode != os.W_OK):
                with self.assertRaisesRegex(ValueError, "own"):
                    self.api.check_tree_access(root, os.geteuid())
                (root / "escape").symlink_to("/etc/passwd")
                with self.assertRaisesRegex(ValueError, "symlink|regular"):
                    self.api.check_tree_access(root, 995)

    def test_writable_inherited_fd_is_rejected_even_through_external_hardlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target.write_bytes(b"synthetic")
            link = Path(tmp) / "external-link"
            os.link(target, link)
            value = target.stat()
            protected = {(value.st_dev, value.st_ino)}
            fd = os.open(link, os.O_WRONLY)
            try:
                with self.assertRaisesRegex(ValueError, "descriptor|fd"):
                    self.api.reject_writable_descriptors(protected)
            finally:
                os.close(fd)
            self.api.reject_writable_descriptors(protected)

    def test_bound_inventory_gate_rejects_manifest_and_artifact_changes(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            root.mkdir()
            artifact = root / "engine.so"
            artifact.write_bytes(b"synthetic")
            manifest = Path(tmp) / "inventory.json"
            manifest.write_text(json.dumps({"schema_version": 1, "files": [{
                "path": "engine.so", "size_bytes": 9, "sha256": hashlib.sha256(b"synthetic").hexdigest()}]}))
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with (mock.patch.object(self.api, "read_credentials", return_value=self.credentials),
                  mock.patch.object(self.api.os, "access", side_effect=lambda p, mode, **kw: mode != os.W_OK)):
                self.assertEqual(self.api.verify_service_tree(root, manifest, digest, 995, 982)["files"], 1)
                with self.assertRaisesRegex(ValueError, "manifest hash"):
                    self.api.verify_service_tree(root, manifest, "0" * 64, 995, 982)
                artifact.write_bytes(b"mutations")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    self.api.verify_service_tree(root, manifest, digest, 995, 982)


if __name__ == "__main__":
    unittest.main()
