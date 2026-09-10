"""Synthetic CPU controls, with the real packaged CLI and native stream parser."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
import importlib.util

HERE = Path(__file__).resolve().parent

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Terminal(unittest.TestCase):
    def setUp(self):
        self.candidate = load(HERE / 'candidate.py', 'terminal_candidate')
        self.base_tests = load(HERE.parent / 'soak-native-001/test_run.py', 'terminal_base_tests')
        self.fixture = self.base_tests.DurabilityEvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.patch.stop()
        self.out = self.fixture.out
        self.api = self.candidate.runner
        manifest = self.api.read(self.out / 'manifest.json')
        manifest['sources'] = {str(p): self.api.probe.sha(p) for p in self.api.SOURCES}
        self.api.probe.write(self.out / 'manifest.json', manifest)
        launch = self.api.read(self.out / 'server-launch.json')
        launch['arguments'][launch['arguments'].index('--max-num-batched-tokens') + 1] = '256'
        launch['arguments'][launch['arguments'].index('--long-prefill-token-threshold') + 1] = '64'
        self.api.probe.write(self.out / 'server-launch.json', launch)
        self.fixture.write_rows(self.out / 'smoke/raw.jsonl', self.fixture.native_rows(600 * 10**9, 'synthetic-startup'))
        smoke = self.api.score_request(self.out / 'smoke/raw.jsonl', self.fixture.ids, self.fixture.fixture)
        smoke.update(verdict='PASS', scope='startup correctness only', manifest_sha256=self.api.verify(self.out))
        self.api.probe.write(self.out / 'smoke/summary.json', smoke)
        self.fixture.journal[0].update(manifest_sha256=self.api.verify(self.out),
            launch_sha256=self.api.probe.sha(self.out / 'server-launch.json'),
            smoke=self.api.validate_smoke(self.out, self.api.verify(self.out), 700 * 10**9))
        self.fixture.write_rows(self.out / 'raw.jsonl', self.fixture.journal)

    def add_window(self):
        window = self.out / 'first-window'
        window.mkdir()
        shutil.copyfile(self.out / 'server-launch.json', window / 'launch.json')
        rows = [{'kind': 'start', 'start_ns': 700 * 10**9, 'deadline_ns': 1000 * 10**9,
            'observed_ns': 700 * 10**9 + 1, 'manifest_sha256': self.api.verify(self.out),
            'launch_sha256': self.api.probe.sha(window / 'launch.json'), 'smoke': self.fixture.journal[0]['smoke']}]
        for index in range(5):
            for worker in range(4):
                name = f'w{worker}-{index:05d}-raw.jsonl'
                self.fixture.write_rows(window / name, self.fixture.native_rows(
                    (701 + 50 * index) * 10**9 + worker * 1000000, f'synthetic-window-{worker}-{index}'))
                result = self.api.score_request(window / name, self.fixture.ids, self.fixture.fixture)
                rows.append({'kind': 'request', 'worker': worker, 'index': index, 'raw_path': name,
                    'verdict': 'PASS', 'observed_ns': result['end_ns'] + 100, **result})
        rows.append({'kind': 'end', 'ended_ns': 951 * 10**9, 'observed_ns': 951 * 10**9 + 1, 'stopped_on_failure': False})
        self.fixture.write_rows(window / 'raw.jsonl', rows)
        self.assertEqual(self.candidate.score_first_window(self.out)['verdict'], 'PASS')
        self.bind()

    def bind(self):
        self.api.probe.write(self.out / 'first-window-admission-binding.json', {
            'scope': 'Completed necessary-condition falsifier before unchanged durability admission',
            'observed_ns': 960 * 10**9, 'manifest_sha256': self.api.verify(self.out),
            'files': {str(p.relative_to(self.out)): self.api.probe.sha(p) for p in sorted((self.out / 'first-window').iterdir())}})

    def cli(self):
        result = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'candidate.py'), 'score', '--output', str(self.out)],
            text=True, capture_output=True, timeout=30)
        summary = self.api.read(self.out / 'summary.json')
        return result.returncode, summary

    def test_missing_prerequisite_cannot_pass_packaged_cli(self):
        code, summary = self.cli()
        self.assertNotEqual(code, 0, summary)
        self.assertEqual(summary['verdict'], 'FAIL')

    def test_valid_bound_native_window_and_full_trace_pass(self):
        self.add_window()
        code, summary = self.cli()
        self.assertEqual(code, 0, summary)
        self.assertEqual(summary['verdict'], 'PASS')

    def test_terminal_mutations_fail_without_repairing_window(self):
        self.add_window()
        files = {p.relative_to(self.out): p.read_bytes() for p in (self.out / 'first-window').iterdir()}
        files[Path('first-window-admission-binding.json')] = (self.out / 'first-window-admission-binding.json').read_bytes()
        for change in ['missing-binding', 'missing-stream', 'extra', 'corrupt-summary', 'corrupt-stream',
                       'stale-valid-window', 'missing-bound-entry', 'extra-bound-entry', 'wrong-manifest',
                       'late-window', 'early-binding', 'late-binding', 'wrong-launch']:
            with self.subTest(change=change):
                window = self.out / 'first-window'
                binding_path = self.out / 'first-window-admission-binding.json'
                binding = self.api.read(binding_path)
                if change == 'missing-binding': binding_path.unlink()
                elif change == 'missing-stream': (window / 'w3-00004-raw.jsonl').unlink()
                elif change == 'extra': (window / 'extra.json').write_text('{}')
                elif change == 'corrupt-summary': (window / 'summary.json').write_text('{}')
                elif change == 'corrupt-stream': (window / 'w0-00000-raw.jsonl').write_text('{}\n')
                elif change == 'stale-valid-window':
                    p = window / 'w0-00000-raw.jsonl'
                    p.write_text(p.read_text().replace('synthetic-window-0-0', 'synthetic-replaced-0-0'))
                    p = window / 'raw.jsonl'
                    p.write_text(p.read_text().replace('synthetic-window-0-0', 'synthetic-replaced-0-0'))
                    self.assertEqual(self.candidate.score_first_window(self.out)['verdict'], 'PASS')
                elif change == 'missing-bound-entry':
                    del binding['files']['first-window/w0-00000-raw.jsonl']
                    self.api.probe.write(binding_path, binding)
                elif change == 'extra-bound-entry':
                    binding['files']['../outside'] = '0' * 64
                    self.api.probe.write(binding_path, binding)
                elif change == 'wrong-manifest':
                    binding['manifest_sha256'] = '0' * 64
                    self.api.probe.write(binding_path, binding)
                elif change in ['early-binding', 'late-binding']:
                    binding['observed_ns'] = (950 if change == 'early-binding' else 1001) * 10**9
                    self.api.probe.write(binding_path, binding)
                elif change == 'late-window':
                    p = window / 'raw.jsonl'
                    rows = [self.api.decode(line) for line in p.read_text().splitlines()]
                    rows[-1].update(ended_ns=1001 * 10**9, observed_ns=1001 * 10**9 + 1)
                    self.fixture.write_rows(p, rows)
                    self.assertEqual(self.candidate.score_first_window(self.out)['verdict'], 'PASS')
                    self.bind()
                elif change == 'wrong-launch':
                    launch = self.api.read(window / 'launch.json')
                    launch['synthetic_identity'] = 'another-launch'
                    self.api.probe.write(window / 'launch.json', launch)
                    p = window / 'raw.jsonl'
                    rows = [self.api.decode(line) for line in p.read_text().splitlines()]
                    rows[0]['launch_sha256'] = self.api.probe.sha(window / 'launch.json')
                    self.fixture.write_rows(p, rows)
                    self.assertEqual(self.candidate.score_first_window(self.out)['verdict'], 'PASS')
                    self.bind()
                summary_before = (window / 'summary.json').read_bytes()
                code, summary = self.cli()
                self.assertNotEqual(code, 0, summary)
                self.assertEqual(summary['verdict'], 'FAIL')
                self.assertEqual((window / 'summary.json').read_bytes(), summary_before)
                shutil.rmtree(window)
                window.mkdir()
                for name, data in files.items(): (self.out / name).write_bytes(data)

    def test_run_completion_uses_terminal_adapter(self):
        self.assertIs(self.api.run.__globals__['score'], self.candidate.score)
        self.add_window()
        (self.out / 'first-window/w0-00000-raw.jsonl').write_text('{}\n')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(self.api.run.__globals__['score'](self.out))

class Transport(unittest.TestCase):
    def test_later_worker_transport_exception_stops_peers(self):
        module = load(HERE / 'candidate.py', 'transport_candidate')
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'api-key').write_text('synthetic-cpu-only')
            entered = threading.Barrier(4)
            calls = []
            def stream(path, *_):
                worker = int(path.name[1])
                calls.append(path.name)
                if path.name[3:8] == '00000': entered.wait(timeout=5)
                if worker == 3: raise OSError('synthetic later-worker transport failure')
                time.sleep(0.05)
                return True
            with patch.object(module.runner, 'verify', return_value='synthetic'), \
                 patch.object(module.runner.probe, 'launch_check', return_value={}), \
                 patch.object(module.runner, 'validate_smoke', return_value={}), \
                 patch.object(module.runner, 'read', return_value={}), \
                 patch.object(module.runner, 'stream', side_effect=stream), \
                 patch.object(module.runner, 'score_request', return_value={}), \
                 patch.object(module, 'score_first_window', return_value={'verdict': 'FAIL'}):
                self.assertEqual(module.first_window(out, out)['verdict'], 'FAIL')
            self.assertEqual(len(calls), 4, calls)
            rows = [json.loads(line) for line in (out / 'first-window/raw.jsonl').read_text().splitlines()]
            self.assertTrue(any(row['kind'] == 'request' and row['worker'] == 3 and row['verdict'] == 'FAIL' for row in rows))
            self.assertTrue(rows[-1]['stopped_on_failure'])

if __name__ == '__main__':
    unittest.main()
