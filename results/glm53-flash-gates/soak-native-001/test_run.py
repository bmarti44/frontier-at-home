"""Synthetic admission/scoring mutations; never model or performance evidence."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('durability_test_target', Path(__file__).with_name('run.py'))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
NANO = 10**9
START = 10**12

class DurabilityEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.patch = mock.patch.object(api, 'SOURCES', [Path(__file__).resolve()])
        self.patch.start()
        self.addCleanup(self.patch.stop)
        values = [f'RECORD_{name}_0123456789abcdef' for name in ['ALPHA', 'BRAVO', 'CHARLIE']]
        self.answer = ', '.join(values + ['NO_EXTRA_RECORD'])
        self.fixture = {'records': [{'case_id': str(i), 'position': 20 + i * 100,
            'value': v, 'expected_sha256': hashlib.sha256(v.encode()).hexdigest()} for i, v in enumerate(values)],
            'absent_value': 'RECORD_DELTA_abcdef0123456789'}
        self.ids = [1] * api.INPUT_TOKENS
        for worker in range(4):
            for name, value in [('request', {'model': 'glm-5.3-flash'}), ('fixture', self.fixture), ('input-token-ids', self.ids)]:
                api.probe.write(self.out / f'{worker}-{name}.json', value)
        api.probe.write(self.out / 'manifest.json', {'configuration': api.CONFIG, 'seed': '0' * 64,
            'files': {name: api.probe.sha(self.out / name) for name in api.FILES},
            'sources': {str(p): api.probe.sha(p) for p in api.SOURCES}})
        launch = {'arguments': ['--max-model-len', '262144', '--max-num-seqs', '4',
            '--max-num-batched-tokens', '128', '--long-prefill-token-threshold', '32', '--no-enable-prefix-caching']}
        api.probe.write(self.out / 'server-launch.json', launch)
        smoke = self.out / 'smoke'
        smoke.mkdir()
        self.write_rows(smoke / 'raw.jsonl', self.native_rows(START - 100 * NANO, 'synthetic-startup'))
        smoke_summary = api.score_request(smoke / 'raw.jsonl', self.ids, self.fixture)
        smoke_summary.update(verdict='PASS', scope='startup correctness only', manifest_sha256=api.verify(self.out))
        api.probe.write(smoke / 'summary.json', smoke_summary)
        entries = []
        for index in range(36):
            for worker in range(4):
                start = START + (1 + index * 50) * NANO + worker * 1000000
                name = f'w{worker}-{index:05d}-raw.jsonl'
                rows = self.native_rows(start, f'synthetic-{worker}-{index}')
                self.write_rows(self.out / name, rows)
                result = api.score_request(self.out / name, self.ids, self.fixture)
                entries.append({'kind': 'request', 'worker': worker, 'index': index,
                    'raw_path': name, 'verdict': 'PASS', 'observed_ns': result['end_ns'] + 100, **result})
        self.journal = [{'kind': 'start', 'observed_ns': START + 1, 'start_ns': START,
            'deadline_ns': START + 1800 * NANO, 'manifest_sha256': api.verify(self.out),
            'smoke': {name: api.probe.sha(smoke / name) for name in ['raw.jsonl', 'summary.json']},
            'launch_sha256': api.probe.sha(self.out / 'server-launch.json')}] + entries + [
            {'kind': 'end', 'drained_ns': START + 1801 * NANO, 'ended_ns': START + 1801200000000,
             'observed_ns': START + 1801200000100, 'stopped_on_failure': False}]
        self.write_rows(self.out / 'raw.jsonl', self.journal)
        body = json.dumps({'data': [{'id': 'glm-5.3-flash', 'max_model_len': 262144}]})
        self.monitor = {'memory': [{'t': i, 'gib': 20.0} for i in range(1802)], 'memory_error': None,
            'sampler_alive_at_stop': True, 'sampler_stopped': True, 'health_thread_stopped': True,
            'health': [{'start_ns': START + i * NANO, 'end_ns': START + i * NANO + 100, 'status': 200, 'body': body}
                for i in [*range(1, 1800, 30), 1801]]}
        api.probe.write(self.out / 'monitor.json', self.monitor)
        self.write_rows(self.out / 'memory.jsonl', [{'observed_ns': START + row['t'] * NANO + 100, 'sample': row} for row in self.monitor['memory']])
        self.write_rows(self.out / 'health.jsonl', [{'observed_ns': row['end_ns'] + 100, 'sample': row} for row in self.monitor['health']])

    def native_rows(self, start, response_id):
        def chunk(offset, choices, **fields):
            return {'kind': 'chunk', 'monotonic_ns': start + offset,
                'chunk': {'id': response_id, 'model': 'glm-5.3-flash', 'choices': choices, **fields}}
        return [
            {'kind': 'start', 'monotonic_ns': start}, {'kind': 'http', 'monotonic_ns': start + 1, 'status': 200},
            chunk(2, [{'index': 0, 'delta': {}}], prompt_token_ids=self.ids),
            chunk(NANO, [{'index': 0, 'delta': {'reasoning': 'synthetic reasoning'}, 'token_ids': [2]}]),
            chunk(49 * NANO, [{'index': 0, 'delta': {'content': self.answer}, 'token_ids': [3]}]),
            chunk(49500000000, [{'index': 0, 'delta': {}, 'token_ids': [4], 'finish_reason': 'stop'}]),
            chunk(49500000001, [], usage={'prompt_tokens': len(self.ids), 'completion_tokens': 3, 'total_tokens': len(self.ids) + 3}),
            {'kind': 'done', 'monotonic_ns': start + 49500000002},
            {'kind': 'end', 'monotonic_ns': start + 49600000000}]

    def write_rows(self, path, rows):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def verdict(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return api.score(self.out)

    def mutate_stream(self, mutate):
        path = self.out / 'w0-00000-raw.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        self.write_rows(path, rows)

    def test_complete_synthetic_trace_passes(self):
        self.assertTrue(self.verdict())

    def test_startup_smoke_cannot_be_missing(self):
        shutil.rmtree(self.out / 'smoke')
        self.assertFalse(self.verdict())

    def test_failed_startup_smoke_is_rejected(self):
        summary = api.probe.read(self.out / 'smoke/summary.json')
        summary['verdict'] = 'FAIL'
        api.probe.write(self.out / 'smoke/summary.json', summary)
        self.assertFalse(self.verdict())

    def test_forged_startup_pass_does_not_replace_raw(self):
        (self.out / 'smoke/raw.jsonl').write_text('{}\n')
        self.assertFalse(self.verdict())

    def test_bad_startup_stops_before_model_requests(self):
        (self.out / 'smoke/raw.jsonl').write_text('{}\n')
        with mock.patch.object(api.probe, 'launch_check', return_value={}), mock.patch.object(api, 'stream') as transport:
            with self.assertRaises((ValueError, AssertionError, KeyError)):
                api.run(self.out, self.out)
        transport.assert_not_called()

    def test_changed_valid_startup_after_admission_is_rejected(self):
        path = self.out / 'smoke'
        self.write_rows(path / 'raw.jsonl', self.native_rows(START - 100 * NANO, 'different-startup'))
        summary = api.score_request(path / 'raw.jsonl', self.ids, self.fixture)
        summary.update(verdict='PASS', scope='startup correctness only', manifest_sha256=api.verify(self.out))
        api.probe.write(path / 'summary.json', summary)
        self.assertFalse(self.verdict())

    def test_unused_overflow_number_is_rejected(self):
        path = self.out / 'raw.jsonl'
        path.write_text(path.read_text().replace('"kind": "start"', '"unused_nonfinite": 1e999, "kind": "start"', 1))
        self.assertFalse(self.verdict())

    def test_incremental_memory_observations_are_required(self):
        (self.out / 'memory.jsonl').unlink()
        self.assertFalse(self.verdict())

    def test_incremental_health_observations_are_required(self):
        (self.out / 'health.jsonl').unlink()
        self.assertFalse(self.verdict())

    def test_sample_is_on_disk_before_terminal_aggregation(self):
        path = self.out / 'partial-samples.jsonl'
        samples = api.JournalSamples(path)
        samples.append({'t': 0, 'gib': 20})
        row = json.loads(path.read_text())
        self.assertEqual(row['sample'], {'t': 0, 'gib': 20})
        self.assertGreater(row['observed_ns'], 0)
        self.assertEqual(samples, [row['sample']])

    def test_truncation_is_rejected(self):
        self.mutate_stream(lambda rows: rows[5]['chunk']['choices'][0].update(finish_reason='length'))
        self.assertFalse(self.verdict())

    def test_reasoning_is_not_a_final_answer(self):
        def change(rows):
            rows[4]['chunk']['choices'][0]['delta'] = {'reasoning': self.answer}
        self.mutate_stream(change)
        self.assertFalse(self.verdict())

    def test_wrong_input_ids_rejected(self):
        self.mutate_stream(lambda rows: rows[2]['chunk']['prompt_token_ids'].__setitem__(0, 99))
        self.assertFalse(self.verdict())

    def test_missing_done_rejected(self):
        self.mutate_stream(lambda rows: rows.pop(-2))
        self.assertFalse(self.verdict())

    def test_invented_or_duplicate_response_identity_rejected(self):
        self.journal[1]['response_id'] = self.journal[2]['response_id']
        self.write_rows(self.out / 'raw.jsonl', self.journal)
        self.assertFalse(self.verdict())

    def test_missing_and_extra_raw_files_rejected(self):
        (self.out / 'unexpected-raw.jsonl').write_text('{}\n')
        self.assertFalse(self.verdict())
        (self.out / 'unexpected-raw.jsonl').unlink()
        (self.out / 'w0-00000-raw.jsonl').unlink()
        self.assertFalse(self.verdict())

    def test_sparse_middle_of_soak_is_rejected(self):
        removed = [r for r in self.journal[1:-1] if r['worker'] == 0 and r['index'] == 10]
        self.journal = [r for r in self.journal if r not in removed]
        for row in removed:
            (self.out / row['raw_path']).unlink()
        self.write_rows(self.out / 'raw.jsonl', self.journal)
        self.assertFalse(self.verdict())

    def test_shortened_admission_interval_rejected(self):
        self.journal[0]['deadline_ns'] -= 90 * NANO
        self.write_rows(self.out / 'raw.jsonl', self.journal)
        self.assertFalse(self.verdict())

    def test_low_memory_and_nonfinite_memory_rejected(self):
        for value in [17.99, float('nan'), float('inf')]:
            self.monitor['memory'][50]['gib'] = value
            (self.out / 'monitor.json').write_text(json.dumps(self.monitor))
            self.write_rows(self.out / 'memory.jsonl', [{'observed_ns': START + row['t'] * NANO + 100, 'sample': row} for row in self.monitor['memory']])
            self.assertFalse(self.verdict())

    def test_health_gap_or_wrong_model_rejected(self):
        baseline = list(self.monitor['health'])
        self.monitor['health'] = baseline[:10] + baseline[15:]
        api.probe.write(self.out / 'monitor.json', self.monitor)
        self.write_rows(self.out / 'health.jsonl', [{'observed_ns': row['end_ns'] + 100, 'sample': row} for row in self.monitor['health']])
        self.assertFalse(self.verdict())
        self.monitor['health'] = baseline
        self.monitor['health'][0]['body'] = '{"data":[{"id":"wrong","max_model_len":262144}]}'
        api.probe.write(self.out / 'monitor.json', self.monitor)
        self.write_rows(self.out / 'health.jsonl', [{'observed_ns': row['end_ns'] + 100, 'sample': row} for row in self.monitor['health']])
        self.assertFalse(self.verdict())

    def test_missing_fixture_binding_rejected(self):
        manifest = api.probe.read(self.out / 'manifest.json')
        del manifest['files']['0-request.json']
        api.probe.write(self.out / 'manifest.json', manifest)
        self.assertFalse(self.verdict())

    def test_admission_after_deadline_creates_no_stream(self):
        path = self.out / 'late-raw.jsonl'
        with mock.patch.object(api.urllib.request, 'urlopen', side_effect=AssertionError('HTTP after deadline')):
            self.assertFalse(api.stream(path, {}, 'synthetic-test-only', 1))
        self.assertFalse(path.exists())

if __name__ == '__main__':
    unittest.main()
