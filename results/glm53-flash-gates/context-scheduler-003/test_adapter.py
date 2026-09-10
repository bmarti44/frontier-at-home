"""Direct-context configuration acceptance; no model or network operations."""
import importlib.util
from pathlib import Path
import unittest

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

if __name__ == '__main__':
    unittest.main()
