"""Direct-context configuration acceptance; no model or network operations."""
import importlib.util
import ast
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class Configuration(unittest.TestCase):
    def test_current_profile_and_rejections(self):
        target = HERE / 'adapter.py'
        api = load(target if target.exists() else ROOT / 'scripts/48_probe_glm53_context.py', 'direct_adapter_test')
        launch = {'arguments': ['--max-model-len', '262144', '--max-num-seqs', '4',
                  '--max-num-batched-tokens', '512', '--long-prefill-token-threshold', '128',
                  '--no-enable-prefix-caching']}
        self.assertEqual(api.check_launch(launch), launch)
        for flag, value in [('--max-model-len', '1048576'), ('--max-num-seqs', '3'),
                            ('--max-num-batched-tokens', '128'), ('--long-prefill-token-threshold', '32')]:
            argv = launch['arguments'].copy();argv[argv.index(flag) + 1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                api.check_launch({'arguments': argv})
        for extra in [['--max-num-seqs', '4'], ['--enable-prefix-caching'], [None]]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                api.check_launch({'arguments': launch['arguments'] + extra})

class Reuse(unittest.TestCase):
    def setUp(self):
        self.api = load(HERE / 'adapter.py', 'direct_adapter_reuse')

    def test_closed_functions_and_selected_validator(self):
        old = load(ROOT / 'scripts/48_probe_glm53_context.py', 'old_direct_probe')
        for name in ['verify', 'run', 'score', 'parse_stream', 'launch_check']:
            self.assertEqual(inspect.getsource(getattr(self.api.probe, name)), inspect.getsource(getattr(old, name)))
        self.assertIs(self.api.probe.check_launch, self.api.scheduler.check_launch)
        self.assertIs(self.api.prepare.probe, self.api.probe)
        self.assertIs(self.api.short.probe, self.api.probe)
        for module, path, names in [
            (self.api.prepare, 'prepare_inputs.py', ['prepare']),
            (self.api.short, 'run_short.py', ['verify_inputs', 'main'])]:
            original = load(HERE.parent / 'context-clear-instruction-001' / path, 'original_' + path)
            for name in names:self.assertEqual(inspect.getsource(getattr(module, name)), inspect.getsource(getattr(original, name)))

    def test_binding_statements_unchanged(self):
        old = ast.parse((HERE.parent / 'context-native-profile-002/bind_launch.py').read_text()).body
        start = next(i for i, n in enumerate(old) if isinstance(n, ast.Assign) and
                     any(isinstance(t, ast.Name) and t.id == 'manifest' for t in n.targets))
        new = ast.parse(inspect.getsource(self.api.bind_launch)).body[0].body
        self.assertEqual([ast.dump(n) for n in old[start:]], [ast.dump(n) for n in new])

    def test_frozen_source_coverage_and_rejections(self):
        rows = [{'path': str(p), 'size_bytes': p.stat().st_size, 'sha256': self.api.probe.sha(p)} for p in self.api.SOURCES]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def check(records):
                (root / 'manifest.json').write_text(json.dumps({'files': records}))
                self.api.verify_freeze(root)
            check(rows)
            for changed in [rows[1:], rows + [rows[0]], [{**rows[0], 'sha256': '0' * 64}] + rows[1:]]:
                with self.assertRaises(ValueError):check(changed)

    def test_short_validates_launch_before_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);server = root / 'server';server.mkdir()
            argv = ['adapter.py', 'short', '--frozen', str(root), '--output', str(root), '--server', str(server)]
            with patch.object(self.api, 'verify_freeze'), patch.object(self.api.short, 'main', return_value=0) as request, patch.object(self.api.sys, 'argv', argv):
                with self.assertRaises(FileNotFoundError):self.api.main()
                request.assert_not_called()
                arguments = ['--max-model-len', '262144', '--max-num-seqs', '4', '--max-num-batched-tokens', '128',
                             '--long-prefill-token-threshold', '32', '--no-enable-prefix-caching']
                (server / 'launch.json').write_text(json.dumps({'arguments': arguments}))
                with self.assertRaises(ValueError):self.api.main()
                request.assert_not_called()
                arguments[arguments.index('--max-num-batched-tokens') + 1] = '512'
                arguments[arguments.index('--long-prefill-token-threshold') + 1] = '128'
                (server / 'launch.json').write_text(json.dumps({'arguments': arguments}))
                self.assertEqual(self.api.main(), 0)
                request.assert_called_once_with(root / 'short-correctness', server)

if __name__ == '__main__':
    unittest.main()
