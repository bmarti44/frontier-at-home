"""Exercise the actual worker against retained CPU heap and startup contracts."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[3] / 'scripts/lib/glm53_worker.py'
FLAG = 'GLM53_TRIM_STARTUP_HEAP'


class HeapTrimTests(unittest.TestCase):
    def load(self, setting=None, require_release=False, parent_error=False, trim_releases=True):
        events = []
        state = {'held_kib': 65536, 'live': object()}
        budget = object()

        def collect(): events.append('gc')
        def empty_cache(): events.append('cuda-cache')
        def trim(pad):
            self.assertEqual(pad, 0)
            self.assertEqual(events[-1], 'rss')
            events.append('trim')
            if trim_releases: state['held_kib'] = 0
            return int(trim_releases)

        class Parent:
            def determine_available_memory(self):
                events.append('profile')
                if parent_error: raise RuntimeError('parent profiling failed')
                if require_release and state['held_kib']:
                    raise MemoryError('unused CPU heap still retained before profile')
                state['held_kib'] += 32768
                return budget

        def cdll(name):
            self.assertIsNone(name)
            events.append('bind-libc')
            return types.SimpleNamespace(malloc_trim=trim)

        def proc_open(path, *args, **kwargs):
            self.assertEqual(str(path), '/proc/self/status')
            events.append('rss')
            rss = 100000 + state['held_kib']
            return io.StringIO(f'VmRSS:\t{rss} kB\nRssAnon:\t{rss - 1000} kB\n')

        modules = {
            'ctypes': types.SimpleNamespace(CDLL=cdll, c_size_t=object(), c_int=object()),
            'torch': types.SimpleNamespace(accelerator=types.SimpleNamespace(empty_cache=empty_cache)),
            'vllm.v1.worker.gpu_worker': types.SimpleNamespace(Worker=Parent),
        }
        env = dict(os.environ)
        env.pop(FLAG, None)
        env.pop('GLM53_RELEASE_WARMUP_CACHE', None)
        env['GLM53_RELEASE_LOAD_CACHE'] = '1'
        if setting is not None: env[FLAG] = setting
        with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, env, clear=True):
            spec = importlib.util.spec_from_file_location('heap_trim_worker_test', SOURCE)
            api = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api)
        api.gc = types.SimpleNamespace(collect=collect)
        return api, events, state, budget, proc_open

    def call(self, api, proc_open):
        output = io.StringIO()
        with mock.patch('builtins.open', proc_open), contextlib.redirect_stdout(output):
            result = api.WarmupCleanupWorker().determine_available_memory()
        return result, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_enabled_releases_unused_heap_before_and_after_parent(self):
        api, events, state, budget, proc_open = self.load('1', require_release=True)
        live = state['live']
        result, rows = self.call(api, proc_open)
        self.assertIs(result, budget)
        self.assertIs(state['live'], live)
        self.assertEqual(state['held_kib'], 0)
        self.assertEqual([r['phase'] for r in rows], ['before-profile', 'after-profile'])
        self.assertEqual([r['before']['anonymous_rss_kib'] - r['after']['anonymous_rss_kib'] for r in rows], [65536, 32768])
        self.assertTrue(all(r['event'] == 'glm53_trim_startup_heap' and r['trim_result'] == 1 for r in rows))
        self.assertTrue(all(r['pid'] == os.getpid() and r['start_unix'] <= r['end_unix'] for r in rows))
        self.assertEqual(events.count('gc'), 2)
        self.assertEqual(events.count('trim'), 2)

    def test_default_and_nonexact_flags_do_no_trim_observation_or_binding(self):
        for setting in [None, '0', 'true', '01']:
            with self.subTest(setting=setting):
                api, events, state, budget, proc_open = self.load(setting)
                result, rows = self.call(api, proc_open)
                self.assertIs(result, budget)
                self.assertEqual(rows, [])
                self.assertEqual(events, ['gc', 'cuda-cache', 'profile', 'gc', 'cuda-cache'])

    def test_flag_is_resolved_once_at_initialization(self):
        for initial, later, count in [('1', '0', 2), ('0', '1', 0)]:
            api, events, state, budget, proc_open = self.load(initial)
            with mock.patch.dict(os.environ, {FLAG: later}):
                self.assertIs(self.call(api, proc_open)[0], budget)
            self.assertEqual(events.count('trim'), count)

    def test_parent_failure_propagates_without_post_profile_trim(self):
        api, events, state, budget, proc_open = self.load('1', parent_error=True)
        with self.assertRaisesRegex(RuntimeError, 'parent profiling failed'):
            self.call(api, proc_open)
        self.assertEqual(events.count('trim'), 1)

    def test_zero_release_is_recorded_without_claiming_a_saving(self):
        api, events, state, budget, proc_open = self.load('1', trim_releases=False)
        result, rows = self.call(api, proc_open)
        self.assertIs(result, budget)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['trim_result'], 0)
            self.assertEqual(row['before'], row['after'])

    def test_experimental_profile_selects_only_second_reclamation_variant(self):
        root = SOURCE.parents[2] / 'configs/profiles/glm-5.3-flash'
        experimental = json.loads((root / 'cuda-spark-128g-1m-experimental.json').read_text())
        self.assertEqual(experimental['launch']['env'].get(FLAG), '1')
        self.assertNotIn('GLM53_RELEASE_WARMUP_CACHE', experimental['launch']['env'])
        for name in ['cuda-spark-128g-agent-fast.json', 'cuda-spark-128g-1m.json']:
            self.assertNotIn(FLAG, json.loads((root / name).read_text())['launch']['env'])


if __name__ == '__main__': unittest.main()
