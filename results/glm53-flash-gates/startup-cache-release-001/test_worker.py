"""Exercise the actual startup worker method with a retained-allocation falsifier."""
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[3] / 'scripts/lib/glm53_worker.py'
FLAG = 'GLM53_RELEASE_LOAD_CACHE'

class StartupCleanupTests(unittest.TestCase):
    def load(self, setting=None, require_release=False):
        events = []
        retained = {'live': object(), 'released': False}
        budget = object()
        def collect(): events.append('gc')
        def release():
            events.append('release')
            retained['released'] = True
        class Parent:
            def determine_available_memory(self):
                events.append('profile')
                if require_release and not retained['released']:
                    raise MemoryError('unused load allocation still retained before profile')
                return budget
        modules = {'torch': types.SimpleNamespace(accelerator=types.SimpleNamespace(empty_cache=release)),
                   'vllm.v1.worker.gpu_worker': types.SimpleNamespace(Worker=Parent)}
        env = dict(os.environ)
        env.pop(FLAG, None)
        if setting is not None: env[FLAG] = setting
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location('worker_under_test', SOURCE)
            api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api)
        api.gc = types.SimpleNamespace(collect=collect)
        return api, events, retained, budget

    def test_enabled_closes_retained_load_allocation_failure(self):
        api, events, retained, budget = self.load('1', require_release=True)
        live = retained['live']
        self.assertIs(api.WarmupCleanupWorker().determine_available_memory(), budget)
        self.assertEqual(events, ['gc', 'release', 'profile', 'gc', 'release'])
        self.assertIs(retained['live'], live)

    def test_default_and_explicit_off_preserve_existing_order(self):
        for flag in [None, '0']:
            with self.subTest(flag=flag):
                api, events, retained, budget = self.load(flag)
                self.assertIs(api.WarmupCleanupWorker().determine_available_memory(), budget)
                self.assertEqual(events, ['profile', 'gc', 'release'])

    def test_choice_is_fixed_at_initialization(self):
        api, events, _, _ = self.load('1', require_release=True)
        with mock.patch.dict(os.environ, {FLAG: '0'}):
            api.WarmupCleanupWorker().determine_available_memory()
        self.assertEqual(events, ['gc', 'release', 'profile', 'gc', 'release'])
        api, events, _, _ = self.load('0')
        with mock.patch.dict(os.environ, {FLAG: '1'}):
            api.WarmupCleanupWorker().determine_available_memory()
        self.assertEqual(events, ['profile', 'gc', 'release'])

if __name__ == '__main__': unittest.main()
