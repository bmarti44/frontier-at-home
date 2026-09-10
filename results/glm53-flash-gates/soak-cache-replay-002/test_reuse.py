"""Exercise the real launcher's prepared-state copy against committed evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
class PreparedReplay(unittest.TestCase):
    def test_two_verified_additions_are_reused(self):
        spec = importlib.util.spec_from_file_location('replay_launcher', ROOT / 'scripts/47_run_glm53_dev.py')
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        summary = json.loads((HERE.parent / 'soak-native-007/summary.json').read_text())
        rows = [row for row in summary['compiled_inputs']['added'] if row['path'] not in
            {'.config/vllm/usage_stats.json', '.humming/tmp/lock/launcher.lock'}]
        self.assertEqual(len(rows), 2)
        source = Path.home() / '.cache/glm53-flash/server-20260910-065113/state'
        for row in rows:
            self.assertEqual(hashlib.sha256((source / row['path']).read_bytes()).hexdigest(), row['sha256'])
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'state'
            api.reuse_prepared_kernels(target)
            missing = [row['path'] for row in rows if not (target / row['path']).exists()]
            self.assertEqual(missing, [], 'two recorded prepared files were not copied')
            for row in rows:
                self.assertEqual(hashlib.sha256((target / row['path']).read_bytes()).hexdigest(), row['sha256'])
if __name__ == '__main__': unittest.main()
