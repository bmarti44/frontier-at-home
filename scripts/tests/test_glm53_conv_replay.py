"""Plain JIT replay is sealed without inventing unrelated autotune inputs."""
import importlib
import importlib.util
from pathlib import Path
import shutil
import unittest
import test_glm53_mla_replay as parent


class ConvReplayTests(parent.MLAReplayTests):
    def setUp(self):
        super().setUp()
        self.api = importlib.import_module('glm53_conv_replay')
        shutil.rmtree(self.source / '.cache')
        library = self.source / 'triton/EFGH2345/cuda_utils.so'
        library.parent.mkdir(); library.write_bytes(b'synthetic native helper')
        self.refresh()

    def refresh(self):
        from glm53_mla_replay import file_inventory
        self.record = {'root': str(self.source), 'entries': [dict(type='file', **r) for r in file_inventory(self.source)['files']]}

    def test_unexpected_autotune_or_nontriton_input_rejected(self):
        for path in (self.source / 'triton/ABCD2345/kernel.autotune.json', self.source / 'unlisted.so'):
            path.write_text('{}'); self.refresh()
            with self.assertRaises(ValueError): self.prepare()
            self.assertFalse(self.target.exists()); path.unlink()

    def test_sealed_confirmation_verdict_and_receipt_required(self):
        s = importlib.util.spec_from_file_location('conv_replay_controller', Path(__file__).resolve().parents[1] / '39_run_glm53_probe.py')
        runner = importlib.util.module_from_spec(s); s.loader.exec_module(runner)
        self.assertEqual(runner.probe_verdict('conv-replay', None), 'PASS')
        self.assertEqual(runner.probe_verdict('conv-replay', 'failure'), 'FAIL')
        bundle = {'root': '/sealed', 'sha256': 'a' * 64}
        good = {'selection': 'sealed_convolution_replay', 'bundle': bundle, 'triton_cache_root': '/sealed/triton', 'retuning': 'rejected', 'time_unix': 101.}
        runner.validate_conv_receipt(good, bundle, 102.)
        for key, value in (('bundle', {}), ('retuning', 'allowed'), ('time_unix', 103.), ('time_unix', float('nan')), ('selection', 'preparation')):
            with self.assertRaises(ValueError): runner.validate_conv_receipt(dict(good, **{key: value}), bundle, 102.)


if __name__ == '__main__': unittest.main()
