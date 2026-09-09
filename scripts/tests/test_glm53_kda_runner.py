"""KDA preparation cannot become a binary or context qualification verdict."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('kda_controller_test', Path(__file__).resolve().parents[1] / '39_run_glm53_probe.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class KDARunnerTests(unittest.TestCase):
    def test_kda_preparation_success_stays_no_result(self):
        self.assertEqual(runner.probe_verdict('kda', None), 'NO_RESULT')
        self.assertEqual(runner.probe_verdict('kda', 'failed kernel'), 'FAIL')

    def test_kda_scorer_rejects_missing_or_malformed_rows(self):
        for rows in ([], [{'event': 'configured'}], [{}] * 11):
            with self.assertRaises(ValueError): runner.validate_kda_rows(Path('/nonexistent'), rows, 123)

    def test_complete_bundle_and_mutated_order_or_output_references(self):
        from test_glm53_kda_probe import KDAProbeTests
        fixture = KDAProbeTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        root = fixture.root; seed = 123; api = fixture.api
        rows = [{'event': 'configured', 'case_order': api.case_order(seed), 'request_order': api.request_order(seed),
                 'state_shape': [5, 64, 128, 128], 'pinned_staging': True,
                 'pinned_staging_scope': 'probe_owned_buffers_only'}]
        for count in api.case_order(seed):
            paths, digests = fixture.capture(count, seed)
            names = [f'output-{count}.bf16.gz', f'state-{count}.fp32.gz']
            for path, name in zip(paths, names): path.rename(root / name)
            rows.extend([{'event': 'start', 'query_rows': count},
                         {'event': 'output', 'query_rows': count,
                          'artifacts': [{'file': name, 'sha256': digest} for name, digest in zip(names, digests)],
                          'cuda_elapsed_ms': 1.0, 'cuda_peak_allocated': 30000000, 'cuda_memory_reserved': 40000000}])
        for index, row in enumerate(rows): row['time_unix'] = 1700000000 + index
        for name in ('manifest.json', 'summary.json', 'raw.jsonl', 'traceback.log'): (root / name).write_text('{}')
        checks = runner.validate_kda_rows(root, rows, seed)
        self.assertEqual(len(checks), 5)
        for index, key, value in ((0, 'pinned_staging', 1), (0, 'pinned_staging_scope', 'all_host_device_copies'),
                                   (0, 'request_order', [0, 0, 0, 0]),
                                   (1, 'query_rows', 0), (2, 'cuda_elapsed_ms', float('inf')),
                                   (2, 'artifacts', [{'file': '../output.gz', 'sha256': 'a' * 64}])):
            changed = json.loads(json.dumps(rows)); changed[index][key] = value
            with self.assertRaises(ValueError): runner.validate_kda_rows(root, changed, seed)


if __name__ == '__main__': unittest.main()
