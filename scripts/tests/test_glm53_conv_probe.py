"""Independent scalar convolution check and malformed evidence rejection."""
import gzip
import hashlib
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


class ConvProbeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('conv_test', Path(__file__).resolve().parents[1] / '44_probe_glm53_conv.py')
        self.api = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.api)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_scalar_tap_order_and_short_history(self):
        import numpy as np
        a = self.api; seed = 123
        inputs, weights, initial, ids, fresh = a.fixture(seed, 'prefill-ragged', channels=16)
        output, final = a.expected(seed, 'prefill-ragged', channels=16)
        starts = [0, 1, 8, 519]
        for request, start in enumerate(starts):
            slot = ids[request]
            for c in (0, 3, 15):
                history = list(initial[slot, :, c]) if fresh[request] else [0., 0., 0.]
                for offset in range(min(2, a.CASES['prefill-ragged'][request])):
                    values = history + [float(inputs[start + offset, c])]
                    z = sum(float(x) * float(w) for x, w in zip(values, weights[c]))
                    self.assertAlmostEqual(float(output[start + offset, c]), z / (1 + math.exp(-z)), places=7)
                    history = values[1:]
            if request == 0:
                np.testing.assert_array_equal(final[slot, :, 0], [initial[slot, 1, 0], initial[slot, 2, 0], inputs[0, 0]])
        np.testing.assert_array_equal(initial[0], final[0])
        for invalid in (-1, 2**64, True, 1.5):
            with self.assertRaises(ValueError): a.case_order(invalid)

    def capture(self, case='decode-2', mutate=None):
        a = self.api; seed = 123; channels = 16
        source, _, _, _, _ = a.fixture(seed, case, channels)
        target, state = a.expected(seed, case, channels)
        output = a.bf16_bytes(target); state = a.bf16_bytes(state)
        post = bytearray(a.bf16_bytes(source))
        if case.startswith('decode'):
            for i in range(sum(a.CASES[case])):
                post[i*(channels+320)*2:i*(channels+320)*2+channels*2] = output[i*channels*2:(i+1)*channels*2]
        payload = [output, state, bytes(post)]
        if mutate: mutate(payload)
        paths = [self.root / f'{name}.gz' for name in ('output', 'state', 'input')]
        for path, blob in zip(paths, payload): path.write_bytes(gzip.compress(blob))
        return paths, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]

    def test_every_output_state_and_input_byte_is_scored(self):
        for case in ('decode-2', 'prefill-ragged', 'prefill-ragged-fresh'):
            paths, hashes = self.capture(case)
            result = self.api.score_tensors(paths, hashes, 123, case, channels=16)
            self.assertEqual(result['mismatched_elements'], 0)
            for index in range(3):
                def mutate(blobs, index=index): blobs[index] = b'\xff\xff' + blobs[index][2:]
                paths, hashes = self.capture(case, mutate)
                with self.assertRaises(ValueError): self.api.score_tensors(paths, hashes, 123, case, channels=16)
            paths, hashes = self.capture(case)
            paths[0].write_bytes(paths[0].read_bytes() + gzip.compress(b'extra'))
            hashes[0] = hashlib.sha256(paths[0].read_bytes()).hexdigest()
            with self.assertRaises(ValueError): self.api.score_tensors(paths, hashes, 123, case, channels=16)

    def test_complete_capture_semantic_mutations(self):
        import copy
        from unittest.mock import patch
        a = self.api; seed = 123
        with patch.object(a, 'CHANNELS', 16):
            rows = [{'time_unix': 100., 'event': 'configured', 'case_order': a.case_order(seed), 'slots': a.slots(seed),
                     'pinned_staging': True, **a.geometry()}]
            for case in a.case_order(seed):
                paths, hashes = self.capture(case)
                names = [f'{kind}-{case}.bf16.gz' for kind in ('output', 'state', 'input')]
                for p, name in zip(paths, names): p.rename(self.root / name)
                decode = case.startswith('decode')
                rows.extend([{'time_unix': 100. + len(rows), 'event': 'start', 'case': case},
                             {'time_unix': 101. + len(rows), 'event': 'output', 'case': case,
                              'artifacts': [{'file': n, 'sha256': h} for n, h in zip(names, hashes)],
                              'cuda_elapsed_ms': 1., 'cuda_peak_allocated': 100000000, 'cuda_memory_reserved': 120000000,
                              'input_stride': [336, 1], 'state_stride': [48, 1, 16], 'output_alias': decode,
                              'metadata': None if decode else {'programs': sum((n+7)//8 for n in a.CASES[case]),
                                  'pinned': True, 'gpu_shapes': [[2048], [2048]], 'host_dtypes': ['torch.int64', 'torch.int32']}}])
            for name in ('manifest.json', 'summary.json', 'raw.jsonl', 'traceback.log'): (self.root/name).write_text('{}')
            self.assertEqual(len(a.score_capture(self.root, rows, seed)), 8)
            for index, key, value in ((0, 'slots', [1,1,1,1]), (0, 'pinned_staging', 1), (1, 'case', 'missing'),
                                      (2, 'cuda_peak_allocated', 0), (2, 'cuda_elapsed_ms', float('nan')),
                                      (2, 'output_alias', not rows[2]['output_alias']), (2, 'state_stride', [48,3,1]),
                                      (2, 'artifacts', [{'file': '../escape', 'sha256': 'a'*64}]),
                                      (2, 'time_unix', 0)):
                changed = copy.deepcopy(rows); changed[index][key] = value
                with self.assertRaises(ValueError): a.score_capture(self.root, changed, seed)
            prefill = next(i for i,r in enumerate(rows) if r.get('metadata'))
            changed = copy.deepcopy(rows); changed[prefill]['metadata']['pinned'] = False
            with self.assertRaises(ValueError): a.score_capture(self.root, changed, seed)
            with self.assertRaises(ValueError): a.score_capture(self.root, rows[:-1], seed)

    def test_preparation_verdict_and_missing_capture_rejected(self):
        spec = importlib.util.spec_from_file_location('conv_runner', Path(__file__).resolve().parents[1] / '39_run_glm53_probe.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        self.assertEqual(runner.probe_verdict('conv', None), 'NO_RESULT')
        self.assertEqual(runner.probe_verdict('conv', 'failure'), 'FAIL')
        with self.assertRaises(ValueError): self.api.score_capture(self.root, [], 123)


if __name__ == '__main__': unittest.main()
