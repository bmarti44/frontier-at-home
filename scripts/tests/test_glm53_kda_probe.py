"""CPU analytic KDA reference and malformed-output rejection."""
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


class KDAProbeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('kda_probe_test', Path(__file__).resolve().parents[1] / '41_probe_glm53_kda.py')
        self.api = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.api)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def capture(self, count=2, seed=123, mutation=None):
        import numpy as np
        output, state = self.api.expected(seed, count)
        if mutation: mutation(output, state)
        # Synthetic BF16 round-to-nearest-even fixtures; never qualification data.
        bits = output.astype('<f4').view('<u4')
        bf16 = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2')
        paths = [self.root / 'output.gz', self.root / 'state.gz']
        for path, data in zip(paths, (bf16.tobytes(), state.astype('<f4').tobytes())):
            path.write_bytes(gzip.compress(data))
        return paths, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

    def test_reference_initial_condition_and_null_slot(self):
        import math
        import numpy as np
        output, state = self.api.expected(123, 1)
        slot = self.api.request_order(123)[0]
        n = 1 / math.sqrt(1 + 1e-6)
        initial = self.api.initial_coefficient(123)
        coefficient = math.exp(-2.5) * initial + 0.5 * (1 - math.exp(-2.5) * initial * n) * n
        self.assertAlmostEqual(float(output[0, 0, 0]), (slot + 1) * 0.5 * coefficient * n / math.sqrt(128), places=7)
        self.assertTrue(np.all(state[0] == 3.25))

    def test_scores_every_output_and_state_element(self):
        paths, digests = self.capture()
        result = self.api.score_tensors(*paths, digests, 2, 123)
        self.assertEqual(result['elements'], 2 * 64 * 128 + 5 * 64 * 128 * 128)
        self.assertEqual(result['mismatched_elements'], 0)

    def test_zero_nonfinite_wrong_slot_and_trailing_data_reject(self):
        mutations = [lambda o, s: o.fill(0), lambda o, s: s.__setitem__((0, 0, 0, 0), 0),
                     lambda o, s: o.__setitem__((0, 0, 0), float('nan'))]
        for mutation in mutations:
            paths, digests = self.capture(mutation=mutation)
            with self.assertRaises(ValueError): self.api.score_tensors(*paths, digests, 2, 123)
        paths, digests = self.capture()
        paths[0].write_bytes(paths[0].read_bytes() + gzip.compress(b'extra'))
        digests[0] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
        with self.assertRaises(ValueError): self.api.score_tensors(*paths, digests, 2, 123)


if __name__ == '__main__': unittest.main()
