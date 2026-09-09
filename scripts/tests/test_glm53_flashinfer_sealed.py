"""Pinned FlashInfer loader control flow, GPU hidden and compiler forbidden."""
from contextlib import ExitStack
import hashlib
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import glm53_runtime_jit as policy


class FlashInferSealedTests(unittest.TestCase):
    def setUp(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'FLASHINFER_CUDA_ARCH_LIST': '12.1a'}):
            self.core = importlib.import_module('flashinfer.jit.core')
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)
        self.path = self.root / 'sparse_mla_sm120/sparse_mla_sm120.so'
        self.path.parent.mkdir(); self.path.write_bytes(b'synthetic shared-library fixture')
        self.manifest = {'schema_version': 1, 'files': [{'path': str(self.path.relative_to(self.root)),
            'size_bytes': self.path.stat().st_size, 'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest()}]}
        self.modules = {'sparse_mla_sm120': 'sparse_mla_sm120/sparse_mla_sm120.so'}
        self.loader = self.stack.enter_context(mock.patch.object(self.core.tvm_ffi, 'load_module', return_value=object()))
        self.stack.enter_context(mock.patch.object(self.core, 'tvm_ffi', self.core.tvm_ffi))
        self.compiler = self.stack.enter_context(mock.patch.object(self.core, 'run_ninja', side_effect=AssertionError('compiler entered')))
        self.writer = self.stack.enter_context(mock.patch.object(self.core.JitSpecNvcc, 'write_ninja', side_effect=AssertionError('build writer entered')))
        self.stack.enter_context(mock.patch.dict(os.environ, {'FLASHINFER_DISABLE_JIT': ''}))
        self.stack.enter_context(mock.patch.object(self.core.jit_env, 'FLASHINFER_AOT_DIR', self.root / 'missing-aot'))
        self.stack.enter_context(mock.patch.object(self.core.jit_env, 'FLASHINFER_JIT_DIR', self.root))
        for obj, name in ((self.core.JitSpec, 'build_and_load'), (self.core.JitSpecNvcc, 'build'),
                          (self.core.JitSpecNvcc, 'try_load'), (self.core.JitSpecNvcc, 'load')):
            self.stack.enter_context(mock.patch.object(obj, name, getattr(obj, name)))

    def spec(self, name='sparse_mla_sm120'):
        return self.core.JitSpecNvcc(name, [], [], [], [], [])

    def select(self, enabled=True):
        # The unchanged runtime exercises the real build-and-load fallback for RED.
        if hasattr(policy, 'activate_flashinfer'):
            return policy.activate_flashinfer(self.root, self.manifest, self.modules, enabled=enabled)

    def test_frozen_warm_hit_loads_once_without_compiler_or_writer(self):
        self.select()
        for _ in range(2): self.assertIs(self.spec().build_and_load(), self.loader.return_value)
        self.loader.assert_called_once_with(str(self.path))
        self.compiler.assert_not_called(); self.writer.assert_not_called()

    def test_disabled_selection_preserves_every_entry_point(self):
        before = (self.core.JitSpec.build_and_load, self.core.JitSpecNvcc.build,
                  self.core.JitSpecNvcc.load, self.core.JitSpecNvcc.try_load, self.core.run_ninja)
        self.select(False)
        self.assertEqual(before, (self.core.JitSpec.build_and_load, self.core.JitSpecNvcc.build,
            self.core.JitSpecNvcc.load, self.core.JitSpecNvcc.try_load, self.core.run_ninja))
        self.loader.assert_not_called()

    def test_disabled_selection_does_not_import_or_inspect_cache(self):
        with mock.patch('builtins.__import__', side_effect=AssertionError('import in disabled selection')), \
                mock.patch.object(policy, 'verify_inventory', side_effect=AssertionError('cache access while disabled')):
            self.select(False)

    def test_missing_or_corrupt_library_rejects_before_loading(self):
        for data in (b'changed bytes', None):
            if data is None: self.path.unlink()
            else: self.path.write_bytes(data)
            with self.assertRaises(ValueError): self.select()
        self.loader.assert_not_called(); self.compiler.assert_not_called()

    def test_load_failure_propagates_without_fallback_or_runtime_mutation(self):
        before = self.core.JitSpec.build_and_load
        self.loader.side_effect = ImportError('unloadable frozen library')
        with self.assertRaisesRegex(ImportError, 'unloadable'): self.select()
        self.assertIs(self.core.JitSpec.build_and_load, before)
        self.compiler.assert_not_called(); self.writer.assert_not_called()

    def test_unlisted_module_explicit_path_and_direct_build_reject(self):
        retained = self.core.JitSpec.build_and_load
        retained_build = self.core.JitSpecNvcc.build
        self.select()
        for call in (lambda: self.spec('unknown').build_and_load(),
                     lambda: retained(self.spec('unknown')),
                     lambda: self.spec().load(self.root / 'outside.so'),
                     lambda: self.spec().build(),
                     lambda: retained_build(self.spec())):
            with self.assertRaisesRegex(ValueError, 'sealed'): call()
        self.compiler.assert_not_called(); self.writer.assert_not_called()

    def test_invalid_bindings_reject_before_loading(self):
        for modules in ({}, {'../escape': str(self.path)}, {'sparse_mla_sm120': '../outside.so'},
                        {'sparse_mla_sm120': 'missing.so'}):
            self.modules = modules
            with self.assertRaises(ValueError): self.select()
        self.loader.assert_not_called()

    def test_retained_load_alias_cannot_load_an_unlisted_binary(self):
        original = self.core.JitSpecNvcc.load
        self.select()
        with self.assertRaisesRegex(ValueError, 'sealed'):
            original(self.spec('unknown'), self.root / 'outside.so')
        self.assertIs(original(self.spec(), self.path), self.loader.return_value)
        self.loader.assert_called_once_with(str(self.path))


if __name__ == '__main__':
    unittest.main()
