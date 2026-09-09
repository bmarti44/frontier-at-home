"""Runtime packaging must close symlink escapes and preserve source bytes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))


class RuntimePackContract(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module("glm53_runtime_pack")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "python"
        self.site = self.root / "venv/lib/python3.12/site-packages"
        self.out = self.root / "candidate"
        for path in (self.base / "bin", self.base / "lib/python3.12/site-packages", self.site):
            path.mkdir(parents=True)
        (self.base / "bin/python3.12").write_bytes(b"synthetic-interpreter")
        (self.base / "bin/python3.12").chmod(0o755)
        (self.base / "bin/python3").symlink_to("python3.12")
        (self.base / "lib/python3.12/os.py").write_bytes(b"synthetic-stdlib")
        (self.base / "lib/python3.12/site-packages/unwanted.py").write_bytes(b"base-only")
        (self.site / "engine.py").write_bytes(b"synthetic-engine")

    def pack(self):
        return self.api.package_runtime(self.base, self.site, self.out)

    def test_copies_closed_prefix_and_selected_packages_without_source_mutation(self):
        result = self.pack()
        prefix = self.out / "runtime"
        self.assertEqual((prefix / "bin/python3").read_bytes(), b"synthetic-interpreter")
        self.assertFalse(any(p.is_symlink() for p in prefix.rglob("*")))
        self.assertFalse((prefix / "pyvenv.cfg").exists())
        self.assertFalse((prefix / "lib/python3.12/site-packages/unwanted.py").exists())
        self.assertTrue((self.base / "bin/python3").is_symlink())
        self.assertEqual((self.site / "engine.py").read_bytes(), b"synthetic-engine")
        manifest = json.loads((self.out / "inventory.json").read_text())
        from glm53_contract import verify_inventory
        verify_inventory(prefix, manifest)
        self.assertEqual(result["qualification"], "packaging_only")
        self.assertEqual(result["inventory_sha256"], hashlib.sha256((self.out / "inventory.json").read_bytes()).hexdigest())

    def test_rejects_external_and_directory_cycle_links_before_output(self):
        for target in (self.site / "engine.py", self.base):
            with self.subTest(target=target):
                link = self.base / "escape"
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, "symlink|cycle|escape"):
                    self.pack()
                self.assertFalse(self.out.exists())
                link.unlink()

    def test_never_merges_or_replaces_existing_candidate(self):
        self.out.mkdir()
        sentinel = self.out / "existing"
        sentinel.write_bytes(b"preserve")
        with self.assertRaises(FileExistsError):
            self.pack()
        self.assertEqual(sentinel.read_bytes(), b"preserve")

    def test_base_site_packages_exclusion_also_applies_through_lib64_alias(self):
        (self.base / "lib64").symlink_to("lib")
        self.pack()
        self.assertFalse((self.out / "runtime/lib64/python3.12/site-packages").exists())
        self.assertEqual((self.out / "runtime/lib64/python3.12/os.py").read_bytes(), b"synthetic-stdlib")

    def test_rejects_output_nested_in_input_before_copy(self):
        self.out = self.site / "candidate"
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.pack()
        self.assertFalse(self.out.exists())

    def test_rejects_special_files_and_venv_prefix(self):
        import os
        fifo = self.site / "pipe"
        os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError, "regular|special"):
            self.pack()
        fifo.unlink()
        (self.base / "pyvenv.cfg").write_text("home = /elsewhere\n")
        with self.assertRaisesRegex(ValueError, "venv"):
            self.pack()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
