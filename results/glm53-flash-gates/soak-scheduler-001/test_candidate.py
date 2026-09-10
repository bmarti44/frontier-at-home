"""Candidate configuration and necessary-window evidence contract; no GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Configuration(unittest.TestCase):
    def setUp(self):
        self.module = load(HERE / 'candidate.py', 'scheduler_candidate') if (HERE / 'candidate.py').exists() else load(ROOT / 'scripts/48_probe_glm53_context.py', 'old_scheduler_check')
        self.launch = {'arguments': ['--max-model-len', '262144', '--max-num-seqs', '4',
            '--max-num-batched-tokens', '256', '--long-prefill-token-threshold', '64',
            '--no-enable-prefix-caching']}

    def test_declared_configuration_is_accepted_without_rewriting(self):
        original = json.dumps(self.launch)
        self.assertEqual(self.module.check_launch(self.launch), self.launch)
        self.assertEqual(json.dumps(self.launch), original)

    def test_other_geometry_and_duplicate_options_are_rejected(self):
        for flag, value in [('--max-model-len', '65536'), ('--max-num-seqs', '3'),
                ('--max-num-batched-tokens', '128'), ('--long-prefill-token-threshold', '32')]:
            changed = json.loads(json.dumps(self.launch))
            changed['arguments'][changed['arguments'].index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises((AssertionError, ValueError)):
                self.module.check_launch(changed)
            changed = json.loads(json.dumps(self.launch))
            changed['arguments'] += [flag, self.launch['arguments'][self.launch['arguments'].index(flag) + 1]]
            with self.subTest(duplicate=flag), self.assertRaises((AssertionError, ValueError)):
                self.module.check_launch(changed)

    def test_prefix_enablement_and_malformed_argv_are_rejected(self):
        for suffix in [['--enable-prefix-caching'], ['--enable-prefix-caching=true'],
                       ['--no-enable-prefix-caching'], [None]]:
            with self.subTest(suffix=suffix), self.assertRaises((AssertionError, ValueError)):
                self.module.check_launch({'arguments': self.launch['arguments'] + suffix})

class Window(unittest.TestCase):
    def setUp(self):
        self.module = load(HERE / 'candidate.py', 'scheduler_window_candidate')
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.window = self.out / 'first-window'
        self.window.mkdir()
        self.start = 10**12
        self.rows = [{'kind': 'start', 'start_ns': self.start, 'deadline_ns': self.start + 300 * 10**9,
                      'manifest_sha256': 'fixture-binding', 'launch_sha256': 'launch-binding',
                      'smoke': {}, 'observed_ns': self.start + 100000000}]
        self.results = {}
        for worker in range(4):
            for index in range(5):
                name = f'w{worker}-{index:05d}-raw.jsonl'
                start = self.start + (index * 50 + 1) * 10**9 + worker * 1000
                result = {'start_ns': start, 'end_ns': start + 49 * 10**9,
                          'first_ns': start + 10**9, 'terminal_ns': start + 48 * 10**9,
                          'response_id': f'worker-{worker}-request-{index}'}
                self.results[name] = result
                (self.window / name).write_text('synthetic unit-test placeholder\n')
                self.rows.append({'kind': 'request', 'worker': worker, 'index': index,
                                  'raw_path': name, 'verdict': 'PASS', **result})
        self.rows[1:] = sorted(self.rows[1:], key=lambda row: row['end_ns'])
        for row in self.rows[1:]: row['observed_ns'] = row['end_ns'] + 1
        self.rows.append({'kind': 'end', 'ended_ns': self.start + 251 * 10**9,
                          'observed_ns': self.start + 251 * 10**9 + 1, 'stopped_on_failure': False})
        (self.window / 'launch.json').write_text('{}')
        self.write()

    def write(self):
        (self.window / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def score(self):
        runner = self.module.runner
        with patch.object(runner, 'verify', return_value='fixture-binding'), \
             patch.object(runner, 'validate_smoke', return_value={}), \
             patch.object(self.module, 'check_launch'), \
             patch.object(runner.probe, 'sha', return_value='launch-binding'), \
             patch.object(runner, 'read', return_value={}), \
             patch.object(runner, 'score_request', side_effect=lambda path, *_: self.results[path.name]):
            return self.module.score_first_window(self.out)

    def test_complete_first_window_and_failed_prerequisites(self):
        self.assertEqual(self.score()['verdict'], 'PASS')
        for change in ['missing', 'duplicate', 'late', 'drain', 'extra', 'early-worker', 'no-overlap', 'nonfinite']:
            original = json.loads(json.dumps(self.rows))
            original_results = json.loads(json.dumps(self.results))
            if change == 'missing': self.rows.pop(1)
            elif change == 'duplicate': self.rows.insert(1, self.rows[1])
            elif change == 'late': self.rows[1]['start_ns'] = self.start + 300 * 10**9
            elif change == 'drain': self.rows[-1]['ended_ns'] = self.start + 901 * 10**9
            elif change == 'extra': (self.window / 'extra-raw.jsonl').write_text('unaccounted')
            elif change == 'early-worker': self.rows[1]['start_ns'] = self.start - 1
            elif change == 'no-overlap':
                for row in self.rows[1:-1]:
                    offset = (row['worker'] * 2 + 1) * 10**9
                    row['first_ns'] = row['start_ns'] + offset
                    row['terminal_ns'] = row['first_ns'] + 10**9
                    self.results[row['raw_path']].update(first_ns=row['first_ns'], terminal_ns=row['terminal_ns'])
            elif change == 'nonfinite': self.rows[-1]['ended_ns'] = float('inf')
            self.write()
            with self.subTest(change=change): self.assertEqual(self.score()['verdict'], 'FAIL')
            (self.window / 'extra-raw.jsonl').unlink(missing_ok=True)
            self.rows, self.results = original, original_results
            self.write()

    def test_failed_falsifier_cannot_admit_durability_requests(self):
        with patch.object(self.module, 'score_first_window', return_value={'verdict': 'FAIL'}), \
             patch.object(self.module.runner, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'falsifier must pass'):
                self.module.run_full(self.out, Path('/unused'))
            run.assert_not_called()

if __name__ == '__main__':
    unittest.main()
