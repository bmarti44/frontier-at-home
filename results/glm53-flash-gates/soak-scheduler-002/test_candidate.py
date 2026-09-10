"""CPU configuration/reuse acceptance and inherited real native-stream controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Configuration(unittest.TestCase):
    def test_declared_512_configuration(self):
        target = HERE / 'candidate.py'
        api = load(target if target.exists() else HERE.parent / 'soak-scheduler-001/candidate.py', 'new_scheduler_config')
        launch = {'arguments': ['--max-model-len','262144','--max-num-seqs','4',
            '--max-num-batched-tokens','512','--long-prefill-token-threshold','128','--no-enable-prefix-caching']}
        self.assertEqual(api.check_launch(launch), launch)
        for flag, value in [('--max-model-len','1048576'),('--max-num-seqs','3'),
                            ('--max-num-batched-tokens','256'),('--long-prefill-token-threshold','64')]:
            changed = json.loads(json.dumps(launch))
            changed['arguments'][changed['arguments'].index(flag)+1] = value
            with self.subTest(flag=flag), self.assertRaises(ValueError): api.check_launch(changed)
        for suffix in [['--max-num-batched-tokens','512'],['--enable-prefix-caching'],[None]]:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                api.check_launch({'arguments': launch['arguments'] + suffix})

class PreparedReuse(unittest.TestCase):
    def test_all_fourteen_prepared_additions_are_reused(self):
        api = load(ROOT / 'scripts/47_run_glm53_dev.py', 'new_scheduler_launcher')
        summary = json.loads((HERE.parent / 'soak-native-004/summary.json').read_text())
        rows = [row for row in summary['compiled_inputs']['added'] if row['path'] not in
            {'.config/vllm/usage_stats.json','.humming/tmp/lock/launcher.lock'}]
        self.assertEqual(len(rows), 14)
        origin = Path.home() / '.cache/glm53-flash/server-20260910-051709/state'
        for row in rows:
            self.assertEqual(hashlib.sha256((origin / row['path']).read_bytes()).hexdigest(), row['sha256'])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'state'
            api.reuse_prepared_kernels(target)
            missing = [row['path'] for row in rows if not (target / row['path']).exists()]
            self.assertEqual(missing, [], 'verified prepared additions were not copied')
            for row in rows:
                self.assertEqual(hashlib.sha256((target / row['path']).read_bytes()).hexdigest(), row['sha256'])

base = load(HERE.parent / 'soak-scheduler-001/test_terminal.py', 'new_scheduler_terminal_controls')
base.HERE = HERE

@unittest.skipUnless((HERE / 'candidate.py').exists(), 'new adapter not yet implemented')
class Terminal(base.Terminal):
    def setUp(self):
        super().setUp()
        launch = self.api.read(self.out / 'server-launch.json')
        for flag,value in [('--max-num-batched-tokens','512'),('--long-prefill-token-threshold','128')]:
            launch['arguments'][launch['arguments'].index(flag)+1] = value
        self.api.probe.write(self.out / 'server-launch.json', launch)
        self.fixture.journal[0]['launch_sha256'] = self.api.probe.sha(self.out / 'server-launch.json')
        self.fixture.write_rows(self.out / 'raw.jsonl', self.fixture.journal)

if __name__ == '__main__': unittest.main()
