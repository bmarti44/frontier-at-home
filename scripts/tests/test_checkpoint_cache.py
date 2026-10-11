"""checkpoint_cache.CheckpointCacheEvictor: evict a checkpoint's page cache during engine startup."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))
import checkpoint_cache


class CheckpointCacheEvictorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                     "config.json", "tokenizer.json"):
            (self.dir / name).write_bytes(b"x" * 4096)

    def tearDown(self):
        self.tmp.cleanup()

    def test_evict_once_drops_only_safetensors_pages(self):
        evictor = checkpoint_cache.CheckpointCacheEvictor(str(self.dir))
        seen = []
        with mock.patch.object(checkpoint_cache.os, "posix_fadvise",
                               side_effect=lambda fd, off, n, advice: seen.append(
                                   (os.readlink(f"/proc/self/fd/{fd}"), off, n, advice))):
            evictor.evict_once()
        self.assertEqual(sorted(seen), [
            (str(self.dir / "model-00001-of-00002.safetensors"), 0, 0, os.POSIX_FADV_DONTNEED),
            (str(self.dir / "model-00002-of-00002.safetensors"), 0, 0, os.POSIX_FADV_DONTNEED)])

    def test_start_evicts_at_once_and_stop_joins_after_a_final_pass(self):
        evictor = checkpoint_cache.CheckpointCacheEvictor(str(self.dir), interval_s=3600)
        with mock.patch.object(checkpoint_cache.os, "posix_fadvise") as fadvise:
            evictor.start()
            self.assertEqual(fadvise.call_count, 2)
            self.assertTrue(evictor.running())
            evictor.stop()
            self.assertFalse(evictor.running())
            self.assertEqual(fadvise.call_count, 4)
        self.assertNotIn(checkpoint_cache.THREAD_NAME, [t.name for t in threading.enumerate()])

    def test_background_passes_repeat_until_stopped(self):
        evictor = checkpoint_cache.CheckpointCacheEvictor(str(self.dir), interval_s=0.01)
        passes = threading.Semaphore(0)
        with mock.patch.object(checkpoint_cache.os, "posix_fadvise", side_effect=lambda *a: passes.release()):
            evictor.start()
            for _ in range(8):
                self.assertTrue(passes.acquire(timeout=5))
            evictor.stop()

    def test_a_directory_without_safetensors_fails_loudly(self):
        empty = self.dir / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            checkpoint_cache.CheckpointCacheEvictor(str(empty))

    def test_a_background_failure_is_raised_by_stop_not_swallowed(self):
        evictor = checkpoint_cache.CheckpointCacheEvictor(str(self.dir), interval_s=0.01)
        calls = {"n": 0}

        def fadvise(*_):
            calls["n"] += 1
            if calls["n"] > 2:
                raise OSError(5, "EIO")

        with mock.patch.object(checkpoint_cache.os, "posix_fadvise", side_effect=fadvise):
            evictor.start()
            evictor._thread.join(timeout=5)
            with self.assertRaises(RuntimeError):
                evictor.stop()

    def test_stop_without_start_is_a_no_op(self):
        checkpoint_cache.CheckpointCacheEvictor(str(self.dir)).stop()



class CheckpointCacheReleaseWorkerLifecycleTest(unittest.TestCase):
    """qwen38fn_worker under stub vLLM/torch modules: start at load, stop after warmup."""

    def run_worker(self, body, warm_up=lambda self: "warm"):
        base = type("Worker", (), {
            "load_model": lambda self, *a, **k: "loaded",
            "compile_or_warm_up_model": warm_up,
            "determine_available_memory": lambda self: 0,
        })
        gpu_worker = types.ModuleType("vllm.v1.worker.gpu_worker")
        gpu_worker.Worker = base
        torch = types.ModuleType("torch")
        torch.accelerator = types.SimpleNamespace(empty_cache=lambda: None)
        stubs = {"torch": torch, "vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"),
                 "vllm.v1.worker": types.ModuleType("vllm.v1.worker"), "vllm.v1.worker.gpu_worker": gpu_worker}
        with mock.patch.dict(sys.modules, stubs):
            for name in ("glm53_worker", "qwen38fn_worker"):
                sys.modules.pop(name, None)
            import qwen38fn_worker
            with mock.patch.object(qwen38fn_worker, "CheckpointCacheEvictor") as evictor_cls:
                worker = qwen38fn_worker.CheckpointCacheReleaseWorker()
                worker.vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(model="/m"))
                body(worker, evictor_cls)

    def test_load_starts_and_warmup_stops_the_evictor(self):
        def body(worker, evictor_cls):
            self.assertEqual(worker.load_model(), "loaded")
            evictor_cls.assert_called_once_with("/m")
            evictor_cls.return_value.start.assert_called_once_with()
            evictor_cls.return_value.stop.assert_not_called()
            self.assertEqual(worker.compile_or_warm_up_model(), "warm")
            evictor_cls.return_value.stop.assert_called_once_with()
        self.run_worker(body)

    def test_a_failed_warmup_still_stops_the_evictor(self):
        def warm_up(self):
            raise RuntimeError("capture failed")

        def body(worker, evictor_cls):
            worker.load_model()
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                worker.compile_or_warm_up_model()
            evictor_cls.return_value.stop.assert_called_once_with()
        self.run_worker(body, warm_up)

    def test_warmup_without_this_load_is_a_no_op(self):
        def body(worker, evictor_cls):
            self.assertEqual(worker.compile_or_warm_up_model(), "warm")
            evictor_cls.assert_not_called()
        self.run_worker(body)


if __name__ == "__main__":
    unittest.main()
