"""Raw-tensor controls for the bounded constant-cache MLA falsifier."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / '40_probe_glm53_mla.py'
SPEC = importlib.util.spec_from_file_location('mla_probe', PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

class MLATensorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'output.bf16.gz'
        self.request_order = [2, 0, 3, 1]
        pattern = [1.5, -0.75, 3.0, -6.0]
        self.data = b''.join(struct.pack('<f', value)[2:] for value in pattern * (64 * 512 // 4))
        self.seal(self.data)

    def seal(self, data):
        self.path.write_bytes(gzip.compress(data, mtime=0))
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_complete_analytic_constant_cache_output(self):
        result = probe.score_tensor(self.path, self.digest, 1, self.request_order)
        self.assertEqual(result['elements'], 32768)
        self.assertEqual(result['maximum_absolute_error'], 0.0)
        self.assertEqual(result['mismatched_elements'], 0)

    def test_zero_wrong_slot_and_nonfinite_outputs_reject(self):
        for value in (0.0, 0.5, float('nan'), float('inf')):
            with self.subTest(value=value):
                self.seal(struct.pack('<f', value)[2:] + self.data[2:])
                with self.assertRaisesRegex(ValueError, 'output'):
                    probe.score_tensor(self.path, self.digest, 1, self.request_order)

    def test_missing_extra_bytes_or_digest_reject(self):
        for data in (self.data[:-2], self.data + b'\0\0'):
            self.seal(data)
            with self.assertRaisesRegex(ValueError, 'size'):
                probe.score_tensor(self.path, self.digest, 1, self.request_order)
        self.seal(self.data)
        with self.assertRaisesRegex(ValueError, 'digest'):
            probe.score_tensor(self.path, '0' * 64, 1, self.request_order)

    def test_case_seed_and_order_are_bound_to_public_seed(self):
        a = probe.case_order(123)
        self.assertEqual(set(a), {1, 2, 3, 4, 2048})
        self.assertEqual(a, probe.case_order(123))
        self.assertNotEqual(probe.fixture_seed(123, 1), probe.fixture_seed(124, 1))
        self.assertNotEqual(probe.fixture_seed(123, 1), probe.fixture_seed(123, 2))
