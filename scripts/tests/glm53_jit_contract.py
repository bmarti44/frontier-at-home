#!/usr/bin/env python3
"""Exercise installed Triton cache control flow with synthetic data, no GPU/codegen."""
import base64
import hashlib
import importlib
import json
from pathlib import Path
import sys
import sysconfig
from types import SimpleNamespace
import unittest
from unittest import mock

from test_glm53_sealed_cache import SealedCacheContract
from glm53_runtime_jit import sealed_cache_class

cc = importlib.import_module("triton.compiler.compiler")
rb = importlib.import_module("triton.runtime.build")


class ActualSourceContract(unittest.TestCase):
    setUp = SealedCacheContract.setUp
    factory = SealedCacheContract.factory

    def test_gpu_cache_hit_and_miss_do_not_enter_compiler_stages(self):
        key = base64.b32encode(hashlib.sha256(b"synthetic-cache-key").digest()).decode().rstrip("=")
        self.directory.rename(self.root / key)
        self.key, self.directory = key, self.root / key
        self.group = self.directory / "__grp__kernel.json"
        self.group.write_text(json.dumps({"child_paths": {
            name: str(self.directory / name) for name in ("kernel.json", "kernel.cubin")}}))
        cache_class = self.factory()
        backend = SimpleNamespace(parse_options=lambda value: SimpleNamespace(), add_stages=mock.Mock())
        target = cc.GPUTarget("cuda", 121, 32)
        for name in ("kernel", "missing"):
            source = cc.ASTSource(SimpleNamespace(__name__=name), {})
            with (mock.patch.object(cc, "make_backend", return_value=backend),
                  mock.patch.object(cc, "get_cache_key", return_value="synthetic-cache-key"),
                  mock.patch.object(cc.knobs.cache, "manager_class", cache_class),
                  mock.patch.object(cc.knobs.compilation, "always_compile", False),
                  mock.patch.object(cc.knobs.compilation, "override", False),
                  mock.patch.object(cc.knobs.compilation, "dump_ir", False),
                  mock.patch.object(cc.knobs.compilation, "listener", None),
                  mock.patch.object(cc.knobs.runtime, "add_stages_inspection_hook", None),
                  mock.patch.object(cc, "CompiledKernel", return_value="cached-handle")):
                if name == "kernel":
                    self.assertEqual(cc.compile(source, target=target, _env_vars={}), "cached-handle")
                else:
                    with self.assertRaisesRegex(ValueError, "unsealed|missing"):
                        cc.compile(source, target=target, _env_vars={})
            backend.add_stages.assert_not_called()

    def test_native_load_failure_cannot_fall_through_to_a_compiler(self):
        path = self.directory / ("helper" + sysconfig.get_config_var("EXT_SUFFIX"))
        path.write_bytes(b"deliberately-unloadable-synthetic-library")
        cache = self.factory()(self.key)
        with (mock.patch.object(rb, "get_cache_manager", return_value=cache),
              mock.patch.object(rb, "_load_module_from_path", side_effect=ImportError("unloadable sealed helper")),
              mock.patch.object(rb, "_build", side_effect=AssertionError("compiler entered")) as compiler):
            # The serving adapter must preserve ImportError instead of trying
            # native compilation. This is genuine RED on unchanged Triton.
            with self.assertRaisesRegex(ImportError, "unloadable"):
                rb.compile_module_from_src("synthetic source", "helper")
            compiler.assert_not_called()


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ActualSourceContract))
    raise SystemExit(0 if result.wasSuccessful() else 1)
