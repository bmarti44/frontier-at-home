"""CPU mutations for the frozen MLA raw-output evidence contract."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
runner = load('mla_controller_tests', SCRIPTS / '39_run_glm53_probe.py')

class MLARunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.checks = self.root / 'checks'; self.checks.mkdir()
        self.seed = 123
        self.order = [1, 2, 3, 4, 2048]; random.Random(self.seed).shuffle(self.order)
        requests = [0, 1, 2, 3]; random.Random(self.seed ^ 0x4D4C41).shuffle(requests)
        self.binding = {'scorer_sha256': 'a' * 64, 'binary_sha256': 'b' * 64,
                        'decision': {'sha256': 'c' * 64}, 'metadata': {'config.json': {'sha256': 'd' * 64}}}
        self.rows = [{'event': 'configured', 'backend': 'FlashInferMLASparseSM120Impl', 'cache_bytes': 687865856,
                      'request_order': requests, 'case_order': self.order, 'addressed_last_position': 262143, 'pinned_staging': True}]
        chunks = [b''.join(struct.pack('<f', value * (r + 1))[2:] for value in (0.5, -0.25, 1.0, -2.0)) * 8192 for r in requests]
        for count in self.order:
            chosen = int.from_bytes(hashlib.sha256(json.dumps([self.seed, count], separators=(',', ':')).encode()).digest()[:8], 'big') % 2**63
            self.rows.append({'event': 'start', 'query_rows': count, 'fixture_seed': chosen, 'query_sha256': 'e' * 64})
            path = self.checks / f'output-{count}.bf16.gz'
            with gzip.open(path, 'wb') as stream:
                for i in range(count): stream.write(chunks[i % 4])
            self.rows.append({'event': 'output', 'query_rows': count, 'file': path.name, 'sha256': runner.sha256_file(path),
                              'uncompressed_bytes': count * 65536, 'cuda_elapsed_ms': 1.0,
                              'cuda_peak_allocated': 800000000, 'cuda_memory_reserved': 900000000})
        for i, row in enumerate(self.rows): row['time_unix'] = 1700000000 + i * 0.1
        self.seal()

    def seal(self):
        (self.checks / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        (self.checks / 'summary.json').write_text(json.dumps({'verdict': 'PASS', 'qualification': 'model_free_MLA_constant_cache_falsifier_only',
            'model_loaded': False, 'actual_input_tokens_processed': 0, 'raw_sha256': runner.sha256_file(self.checks / 'raw.jsonl')}))
        (self.checks / 'manifest.json').write_text(json.dumps({'seed': self.seed, **self.binding}))
        (self.checks / 'traceback.log').write_text('')

    def test_preparatory_mla_cannot_claim_binary_qualification(self):
        self.assertEqual(runner.probe_verdict('mla', None), 'NO_RESULT')
        self.assertEqual(runner.probe_verdict('mla', 'kernel failure'), 'FAIL')
        self.assertEqual(runner.probe_verdict('native', None), 'PASS')
        self.assertEqual(runner.probe_verdict('cache', None), 'PASS')
        with self.assertRaises(ValueError): runner.probe_verdict('unknown', None)

    def test_complete_bundle_scores_every_output(self):
        result = runner.score_inner(self.root, 'mla', self.seed, self.binding)
        self.assertEqual(sum(row['elements'] for row in result['tensor_checks']), (2048 + 1 + 2 + 3 + 4) * 32768)

    def test_changed_rng_geometry_or_case_order_rejects(self):
        original = json.loads(json.dumps(self.rows))
        for index, key, value in ((0, 'cache_bytes', 1), (0, 'addressed_last_position', 1023), (0, 'pinned_staging', False),
                                  (1, 'fixture_seed', 0), (2, 'query_rows', 5), (2, 'file', '../output-1.bf16.gz'),
                                  (2, 'cuda_elapsed_ms', float('inf')), (2, 'uncompressed_bytes', 1)):
            with self.subTest(key=key):
                self.rows = json.loads(json.dumps(original)); self.rows[index][key] = value; self.seal()
                with self.assertRaises(ValueError): runner.score_inner(self.root, 'mla', self.seed, self.binding)

    def test_hash_valid_wrong_tensor_rejects(self):
        row = self.rows[2]; path = self.checks / row['file']
        data = gzip.decompress(path.read_bytes()); path.write_bytes(gzip.compress(b'\0\0' + data[2:]))
        row['sha256'] = runner.sha256_file(path); self.seal()
        with self.assertRaisesRegex(ValueError, 'output'): runner.score_inner(self.root, 'mla', self.seed, self.binding)

    def test_missing_case_rejects(self):
        self.rows = self.rows[:-2]; self.seal()
        with self.assertRaises(ValueError): runner.score_inner(self.root, 'mla', self.seed, self.binding)


class GeneratedCacheInventoryTests(unittest.TestCase):
    def test_files_and_symlinks_are_recorded_without_following_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'kernel.so').write_bytes(b'compiled fixture')
            (root / 'outside').symlink_to('/does/not/exist')
            result = runner.generated_cache_inventory(root)
            self.assertIs(result['frozen_before_execution'], False)
            rows = {row['path']: row for row in result['entries']}
            self.assertEqual(rows['kernel.so']['sha256'], hashlib.sha256(b'compiled fixture').hexdigest())
            self.assertEqual(rows['outside'], {'path': 'outside', 'type': 'symlink', 'target': '/does/not/exist'})

    def test_mutation_during_generated_inventory_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'kernel.so'; path.write_bytes(b'original')
            original = runner.sha256_file
            def change(target):
                digest = original(target); target.write_bytes(b'changed size'); return digest
            with mock.patch.object(runner, 'sha256_file', side_effect=change):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    runner.generated_cache_inventory(root)
