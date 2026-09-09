"""KDA cache inputs retain failed trials; replay never retunes. CPU only."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import test_glm53_mla_replay as parent


class KDAReplayTests(parent.MLAReplayTests):
    def setUp(self):
        super().setUp()
        self.api = importlib.import_module('glm53_kda_replay')
        import shutil
        shutil.rmtree(self.source / '.cache')
        self.data = {'key': [16, 'torch.bfloat16'], 'configs_timings': [
            [dict(kwargs={'BV': 32}, num_warps=2, num_ctas=1, num_stages=s,
                  maxnreg=None, pre_hook=None, ir_override=None), timing]
            for s, timing in ((2, [1., .9, 1.1]), (4, [float('inf')] * 3))]}
        self.tuning = self.source / 'triton/ABCD2345/kernel.autotune.json'
        self.tuning.write_text(json.dumps(self.data))
        self.refresh()

    def refresh(self):
        from glm53_mla_replay import file_inventory
        self.record = {'root': str(self.source), 'entries': [dict(type='file', **r)
                       for r in file_inventory(self.source)['files']]}

    def test_failed_trials_preserved_with_finite_sidecar(self):
        original = self.tuning.read_bytes(); binding = self.prepare()
        self.assertEqual((self.target / 'triton/ABCD2345/kernel.autotune.json').read_bytes(), original)
        manifest = self.api.verify_bundle(self.target, binding)
        receipt = manifest['autotune'][0]
        self.assertEqual(receipt['failed_trial_indices'], [1])
        self.assertEqual(receipt['winner_index'], 0)
        self.assertEqual(receipt['trial_statuses'], ['finite', 'failed_infinite_sentinel'])
        json.dumps(receipt, allow_nan=False)

    def test_invalid_tuning_inputs_rejected(self):
        variants = []
        for timing in ([float('nan')]*3, [-float('inf')]*3, [1., float('inf'), 2.], [0., 1., 2.], [True, 1., 2.]):
            data=copy.deepcopy(self.data); data['configs_timings'][1][1]=timing; variants.append(data)
        data=copy.deepcopy(self.data); data['configs_timings'][0][1]=[float('inf')]*3; variants.append(data)
        data=copy.deepcopy(self.data); data['configs_timings'].append(data['configs_timings'][0]); variants.append(data)
        data=copy.deepcopy(self.data); data['extra']=1; variants.append(data)
        for data in variants:
            with self.subTest(data=data):
                self.tuning.write_text(json.dumps(data))
                with self.assertRaises(ValueError): self.api.autotune_receipt(self.tuning, self.source / 'triton')
        self.tuning.write_text('{"key": [], "key": [1], "configs_timings": []}')
        with self.assertRaises(ValueError): self.api.autotune_receipt(self.tuning, self.source / 'triton')

    def test_effective_config_alias_rejected(self):
        data=copy.deepcopy(self.data)
        first=data['configs_timings'][0]
        alias=copy.deepcopy(first); alias[0]['kwargs']['num_warps']=999
        alias[1]=[2., 1.9, 2.1]
        data['configs_timings'].insert(1,alias)
        self.tuning.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'config'):
            self.api.autotune_receipt(self.tuning, self.source / 'triton')


class RetuningGuardTests(unittest.TestCase):
    def test_disabled_selection_has_no_imports(self):
        api=importlib.import_module('glm53_kda_replay')
        with patch('builtins.__import__', side_effect=AssertionError('disabled import')):
            self.assertIsNone(api.reject_retuning(enabled=False))
        with self.assertRaises(ValueError): api.reject_retuning(enabled=1)

    def test_real_autotuner_bench_guard(self):
        api=importlib.import_module('glm53_kda_replay')
        from triton.runtime.autotuner import Autotuner
        original=Autotuner._bench
        try:
            api.reject_retuning(enabled=True)
            with self.assertRaisesRegex(ValueError, 'retuning'): Autotuner._bench(None)
            # Empty-key and prehook paths bypass disk-cache lookup and must fail.
            from types import SimpleNamespace
            for key, configs in (((), []), ((1,), [SimpleNamespace(pre_hook=lambda: None)])):
                with self.assertRaisesRegex(ValueError, 'retuning'):
                    Autotuner.check_disk_cache(None, key, configs, lambda: Autotuner._bench(None))
            # Exercise the real run() cache-disabled branch, with no kernel call.
            tuner=object.__new__(Autotuner)
            tuner.arg_names=['n']; tuner.keys=['n']; tuner.cache={}; tuner.configs=[object(), object()]
            tuner.cache_results=False; tuner.prune_configs=lambda _: tuner.configs
            with self.assertRaisesRegex(ValueError, 'retuning'): tuner.run(1)
            # Ordinary disk cache miss must propagate without invoking benchmark.
            from triton import Config
            from triton.runtime.jit import JITFunction
            module=importlib.import_module('triton.runtime.autotuner')
            fn=object.__new__(JITFunction); fn.__name__='synthetic'; fn._hash='frozen'
            tuner.fn=fn
            with patch.object(JITFunction, 'cache_key', new_callable=PropertyMock, return_value='frozen'), \
                 patch.object(module, 'get_cache_manager', side_effect=ValueError('unsealed miss')), \
                 patch.object(module, 'triton_key', return_value='frozen'), \
                 patch.object(module, 'get_cache_invalidating_env_vars', return_value={}), \
                 patch.object(module.driver, '_active', SimpleNamespace(get_current_target=lambda: None)), \
                 patch('triton.compiler.compiler.make_backend', return_value=SimpleNamespace(hash=lambda:'backend')):
                with self.assertRaisesRegex(ValueError, 'unsealed miss'):
                    tuner.check_disk_cache((1,), [Config({})], lambda: self.fail('benchmark invoked'))
        finally: Autotuner._bench=original


if __name__ == '__main__': unittest.main()
