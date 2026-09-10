"""Check actual provider mapping and reject changed file identities."""
import importlib.util
import os
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('libc_binding_test', Path(__file__).with_name('bind_libc.py'))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class BindingTests(unittest.TestCase):
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
