"""Frozen MLA replay preparation and binding mutations; no GPU work."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))


class MLAReplayTests(unittest.TestCase):
    def setUp(self):
        self.api = importlib.import_module('glm53_mla_replay')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.cleanup)
        self.root = Path(self.temp.name); self.source = self.root / 'source'; self.target = self.root / 'sealed'
        group = self.source / 'triton/ABCD2345'
        group.mkdir(parents=True)
        (group / 'kernel.cubin').write_bytes(b'synthetic frozen kernel')
        (group / 'kernel.json').write_text('{"name":"kernel"}')
        (group / '__grp__kernel.json').write_text(json.dumps({'child_paths': {
            name: str(group / name) for name in ('kernel.cubin', 'kernel.json')}}))
        library = self.source / '.cache/flashinfer/version/121a/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so'
        library.parent.mkdir(parents=True); library.write_bytes(b'synthetic frozen module')
        self.record = {'root': str(self.source), 'entries': [
            {'path': str(p.relative_to(self.source)), 'type': 'file', 'size_bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(self.source.rglob('*')) if p.is_file()]}

    def cleanup(self):
        for p in self.root.rglob('*'):
            if p.is_dir(): p.chmod(0o700)
        self.temp.cleanup()

    def prepare(self):
        return self.api.prepare_bundle(self.source, self.target, self.record)

    def test_relocation_preserves_binaries_and_seals_group_references(self):
        binding = self.prepare(); self.api.verify_bundle(self.target, binding)
        self.assertEqual((self.target / 'triton/ABCD2345/kernel.cubin').read_bytes(), b'synthetic frozen kernel')
        children = json.loads((self.target / 'triton/ABCD2345/__grp__kernel.json').read_text())['child_paths']
        self.assertEqual(children['kernel.cubin'], str(self.target / 'triton/ABCD2345/kernel.cubin'))
        self.assertTrue(all(p.stat().st_mode & 0o222 == 0 for p in (self.target, *self.target.rglob('*'))))

    def test_changed_preparation_rejected_before_copy(self):
        (self.source / 'triton/ABCD2345/kernel.cubin').write_bytes(b'changed')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse(self.target.exists())

    def test_unbound_group_child_rejected(self):
        path = self.source / 'triton/ABCD2345/__grp__kernel.json'
        path.write_text('{"child_paths":{"kernel.json":"/outside/kernel.json"}}')
        for row in self.record['entries']:
            if row['path'].endswith('__grp__kernel.json'):
                row.update(size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'group'): self.prepare()

    def test_changed_manifest_file_or_write_permission_rejected(self):
        binding = self.prepare(); path = self.target / 'triton/ABCD2345/kernel.cubin'
        path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'writ'): self.api.verify_bundle(self.target, binding)
        path.write_bytes(b'changed'); path.chmod(0o444)
        with self.assertRaises(ValueError): self.api.verify_bundle(self.target, binding)


if __name__ == '__main__': unittest.main()
