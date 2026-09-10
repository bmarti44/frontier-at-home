"""Check actual provider mapping and reject changed file identities."""
import importlib.util
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

spec = importlib.util.spec_from_file_location('libc_binding_test', Path(__file__).with_name('bind_libc.py'))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class BindingTests(unittest.TestCase):
    def test_freezer_command_executes_with_actual_resolved_profile(self):
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / 'scripts/lib'))
        import glm53_profile
        snapshot = glm53_profile.resolve_profile(
            'glm-5.3-flash/cuda-spark-128g-1m-experimental', None,
            Path('/tmp/glm53-libc-binding-cpu-preflight'))
        tree = ast.parse(Path(__file__).with_name('freeze.py').read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'libc'
                                  for target in node.targets))
        command_expression = assignment.value.args[0].args[0]
        command = eval(compile(ast.Expression(command_expression), 'frozen-command', 'eval'),
                       {'snapshot': snapshot, 'binder': Path(__file__).with_name('bind_libc.py')})
        result = json.loads(subprocess.check_output(command, text=True))
        self.assertEqual(result['python'], str(Path(snapshot['binary']).resolve()))
        self.assertEqual(result['provider'], api.describe()['provider'])

    def test_actual_provider_is_mapped_without_trim(self):
        provider = api.describe()['provider']
        self.assertTrue(api.verify_mapping(os.getpid(), provider)['maps'])

    def test_changed_provider_binding_rejected(self):
        provider = api.describe()['provider']
        for key in ['size_bytes', 'device', 'inode', 'sha256']:
            with self.subTest(key=key):
                changed = dict(provider)
                changed[key] = 'invalid' if key == 'sha256' else changed[key] + 1
                with self.assertRaisesRegex(ValueError, 'native provider file changed'):
                    api.verify_mapping(os.getpid(), changed)


if __name__ == '__main__': unittest.main()
