"""Synthetic cache safety tests; never model or context-capability evidence."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


class SealedCacheContract(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("glm53_runtime_jit")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = "ABCD2345"
        self.directory = self.root / self.key
        self.directory.mkdir()
        (self.directory / "kernel.json").write_text('{"name":"synthetic"}')
        (self.directory / "kernel.cubin").write_bytes(b"synthetic-not-executable")
        self.group = self.directory / "__grp__kernel.json"
        self.group.write_text(json.dumps({"child_paths": {
            name: str(self.directory / name) for name in ("kernel.json", "kernel.cubin")}}))

    def factory(self):
        manifest = {"schema_version": 1, "files": [
            {"path": str(p.relative_to(self.root)), "size_bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(self.root.rglob("*")) if p.is_file()]}
        return self.api.sealed_cache_class(self.root, manifest)

    def test_complete_frozen_group_returns_exact_children(self):
        cache = self.factory()(self.key)
        self.assertEqual(cache.get_group("kernel.json"), {
            name: str(self.directory / name) for name in ("kernel.json", "kernel.cubin")})

    def test_unknown_specialization_never_creates_a_directory(self):
        cls = self.factory()
        with self.assertRaisesRegex(ValueError, "unknown|unsealed"):
            cls("UNKNOWN2")
        self.assertFalse((self.root / "UNKNOWN2").exists())

    def test_missing_group_and_native_helper_raise_instead_of_returning_none(self):
        cache = self.factory()(self.key)
        for function, name in ((cache.get_group, "missing.json"), (cache.get_file, "cuda_utils.so")):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsealed|missing"):
                function(name)

    def test_deleted_and_replaced_children_are_rejected(self):
        cache = self.factory()(self.key)
        path = self.directory / "kernel.cubin"
        path.write_bytes(b"different-not-executable")
        with self.assertRaises(ValueError):
            cache.get_group("kernel.json")
        path.unlink()
        with self.assertRaises(ValueError):
            cache.get_group("kernel.json")

    def test_group_cannot_reference_an_unlisted_path(self):
        data = json.loads(self.group.read_text())
        data["child_paths"]["kernel.cubin"] = "/tmp/unbound-kernel.cubin"
        self.group.write_text(json.dumps(data))
        cache = self.factory()(self.key)
        with self.assertRaisesRegex(ValueError, "path|unsealed"):
            cache.get_group("kernel.json")

    def test_group_must_include_requested_metadata(self):
        self.group.write_text(json.dumps({"child_paths": {"kernel.cubin": str(self.directory / "kernel.cubin")}}))
        cache = self.factory()(self.key)
        with self.assertRaisesRegex(ValueError, "metadata"):
            cache.get_group("kernel.json")

    def test_malformed_group_is_rejected(self):
        self.group.write_text('{"child_paths":{},"child_paths":{}}')
        with self.assertRaisesRegex(ValueError, "duplicate|metadata"):
            self.factory()(self.key).get_group("kernel.json")

    def test_writes_overrides_dumps_and_path_escapes_are_unavailable(self):
        cls = self.factory()
        for kwargs in ({"override": True}, {"dump": True}):
            with self.assertRaises(ValueError):
                cls(self.key, **kwargs)
        cache = cls(self.key)
        for operation in (lambda: cache.put(b"data", "kernel.cubin"),
                          lambda: cache.put_group("kernel.json", {}),
                          lambda: cache.get_file("../kernel.cubin")):
            with self.assertRaises(ValueError):
                operation()


if __name__ == "__main__":
    unittest.main()
