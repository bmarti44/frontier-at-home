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



class ReleaseTreeTest(unittest.TestCase):
    """release_tree drops every model file's page cache under the model root."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "models"
        (self.root / "a").mkdir(parents=True)
        (self.root / "b" / "deep").mkdir(parents=True)
        for path in ("a/x.gguf", "b/y.safetensors", "b/deep/z.bin", "top.log"):
            (self.root / path).write_bytes(b"x" * 4096)
        outside = Path(self.tmp.name) / "outside"
        outside.mkdir()
        (outside / "secret.bin").write_bytes(b"x")
        (self.root / "link.bin").symlink_to(outside / "secret.bin")
        (self.root / "linkdir").symlink_to(outside, target_is_directory=True)

    def tearDown(self):
        self.tmp.cleanup()

    def fadvised(self, **kwargs):
        seen = []
        with mock.patch.object(checkpoint_cache.os, "posix_fadvise",
                               side_effect=lambda fd, off, n, advice: seen.append(
                                   (os.readlink(f"/proc/self/fd/{fd}"), off, n, advice))):
            count = checkpoint_cache.release_tree(str(self.root), **kwargs)
        return count, sorted(seen)

    def test_every_regular_file_below_the_root_and_no_symlink(self):
        count, seen = self.fadvised()
        files = sorted(str(self.root / p) for p in ("a/x.gguf", "b/y.safetensors", "b/deep/z.bin", "top.log"))
        self.assertEqual(seen, [(f, 0, 0, os.POSIX_FADV_DONTNEED) for f in files])
        self.assertEqual(count, 4)

    def test_an_unreadable_file_is_skipped_not_fatal(self):
        real_open = os.open

        def guarded(path, flags, *a):
            if str(path).endswith("z.bin"):
                raise PermissionError(13, "denied")
            return real_open(path, flags, *a)

        with mock.patch.object(checkpoint_cache.os, "open", side_effect=guarded):
            count, seen = self.fadvised()
        self.assertEqual(count, 3)
        self.assertNotIn(str(self.root / "b/deep/z.bin"), [s[0] for s in seen])

    def test_a_file_deleted_during_the_walk_is_skipped(self):
        # A download's temporary file can vanish between the walk and the open.
        real_open = os.open

        def vanishing(path, flags, *a):
            if str(path).endswith("y.safetensors"):
                raise FileNotFoundError(2, "gone")
            return real_open(path, flags, *a)

        with mock.patch.object(checkpoint_cache.os, "open", side_effect=vanishing):
            count, seen = self.fadvised()
        self.assertEqual(count, 3)
        self.assertNotIn(str(self.root / "b/y.safetensors"), [s[0] for s in seen])

    def test_other_open_errors_and_fadvise_errors_are_raised(self):
        with mock.patch.object(checkpoint_cache.os, "posix_fadvise", side_effect=OSError(5, "EIO")):
            with self.assertRaises(OSError):
                checkpoint_cache.release_tree(str(self.root))

    def test_the_filesystem_root_and_a_missing_directory_are_refused(self):
        for bad in ("/", "", str(self.root / "missing")):
            with self.subTest(root=bad), self.assertRaises(ValueError):
                checkpoint_cache.release_tree(bad)

    def test_mem_free_reads_meminfo_in_gib(self):
        meminfo = Path(self.tmp.name) / "meminfo"
        meminfo.write_text("MemTotal:       125483008 kB\nMemFree:        20971520 kB\nMemAvailable:   120000000 kB\n")
        self.assertEqual(checkpoint_cache.mem_free_gib(str(meminfo)), 20.0)
        meminfo.write_text("MemTotal:       1 kB\n")
        with self.assertRaises(ValueError):
            checkpoint_cache.mem_free_gib(str(meminfo))


class WorkerHarness:
    """Runs a qwen38fn_worker class under stub vLLM/torch modules."""

    def run_worker(self, body, warm_up=lambda self: "warm", cls="CheckpointCacheReleaseWorker"):
        base = type("Worker", (), {
            "load_model": lambda self, *a, **k: "loaded",
            "compile_or_warm_up_model": warm_up,
            "determine_available_memory": lambda self: 0,
            "initialize_from_config": lambda self, config: "allocated",
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
                worker = getattr(qwen38fn_worker, cls)()
                worker.vllm_config = types.SimpleNamespace(model_config=types.SimpleNamespace(model="/m/flash"))
                worker.cache_config = types.SimpleNamespace(kv_cache_memory_bytes=None)
                self.module = qwen38fn_worker
                body(worker, evictor_cls)


class CheckpointCacheReleaseWorkerLifecycleTest(WorkerHarness, unittest.TestCase):
    """The evictor starts at load and stops after warmup."""

    def test_load_starts_and_warmup_stops_the_evictor(self):
        def body(worker, evictor_cls):
            self.assertEqual(worker.load_model(), "loaded")
            evictor_cls.assert_called_once_with("/m/flash")
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



class HostCacheReleaseWorkerTest(WorkerHarness, unittest.TestCase):
    """Candidate 8: drop other engines' model-file cache, and refuse a KV arena MemFree cannot cover.

    Candidate 7 (results/qwen38fn-gates/qualify-2026-10-11-c7) passed the
    MemAvailable admission with qwen38-1m's GGUF still cached. MemFree was
    10.2 GiB at the 17.9 GiB arena, and the driver logged 38 NVRM lines.
    """

    @staticmethod
    def arena(*sizes):
        return types.SimpleNamespace(kv_cache_tensors=[types.SimpleNamespace(size=s) for s in sizes])

    def host(self, free_gib):
        calls = []
        release = mock.patch.object(self.module, "release_tree", side_effect=lambda root: calls.append(root) or 7)
        free = mock.patch.object(self.module, "mem_free_gib", return_value=free_gib)
        return calls, release, free

    def test_load_releases_the_model_root_before_the_weights_and_the_evictor(self):
        def body(worker, evictor_cls):
            order = []
            with mock.patch.object(self.module, "release_tree",
                                   side_effect=lambda root: order.append(("release", root)) or 7):
                evictor_cls.return_value.start.side_effect = lambda: order.append(("evictor", None))
                self.assertEqual(worker.load_model(), "loaded")
            self.assertEqual(order, [("release", "/m"), ("evictor", None)])
        self.run_worker(body, cls="HostCacheReleaseWorker")

    def test_the_arena_is_allocated_only_when_mem_free_covers_it_plus_the_margin(self):
        gib = 2**30

        def body(worker, evictor_cls):
            margin = self.module.KV_FREE_MARGIN_GIB
            self.assertEqual(margin, 2.0)
            calls, release, free = self.host(17.9 + 1.5 + margin)
            with release, free, self.assertLogs("vllm.qwen38fn_worker", "INFO") as logs:
                # The cache groups' tensors all view one 17.9 GiB backing allocation.
                arena = self.arena(*[int(17.9 * gib)] * 6)
                self.assertEqual(worker.initialize_from_config(arena), "allocated")
            self.assertEqual(calls, ["/m"])
            self.assertIn("MemFree 21.40 GiB", logs.output[0])
            self.assertIn("KV arena 17.90 GiB + margin 2.00 GiB", logs.output[0])
            calls, release, free = self.host(10.2)
            with release, free, self.assertLogs("vllm.qwen38fn_worker", "ERROR"):
                with self.assertRaisesRegex(RuntimeError, "refusing to allocate the KV arena"):
                    worker.initialize_from_config(self.arena(int(17.9 * gib)))
        self.run_worker(body, cls="HostCacheReleaseWorker")

    def test_the_configured_arena_counts_when_larger_than_the_tensors(self):
        def body(worker, evictor_cls):
            worker.cache_config.kv_cache_memory_bytes = 19_220_000_000
            calls, release, free = self.host(19.0)
            with release, free, self.assertRaises(RuntimeError):
                worker.initialize_from_config(self.arena(1))
        self.run_worker(body, cls="HostCacheReleaseWorker")

    def test_declared_release_roots_are_released_before_the_weights_and_the_arena(self):
        """Candidate 10: another runtime's cached libraries sit outside the model root.

        The lever screen's a1 (results/qwen38fn-gates/tune-screen-2026-10-11/ATTEMPTS.md)
        met its arena at MemFree 19.85 GiB, short of 19.90, with runtime-001's
        2.56 GiB of libraries still cached. QWEN38FN_HOST_RELEASE_ROOTS names more
        trees to release: once at import, startup-only, empty by default.
        """
        gib = 2**30

        def body(worker, evictor_cls):
            self.assertEqual(self.module.RELEASE_ROOTS, ("/rt/one", "/rt/two"))
            order = []
            with mock.patch.object(self.module, "release_tree",
                                   side_effect=lambda root: order.append(("release", root)) or 7):
                evictor_cls.return_value.start.side_effect = lambda: order.append(("evictor", None))
                worker.load_model()
            self.assertEqual(order, [("release", "/m"), ("release", "/rt/one"), ("release", "/rt/two"),
                                     ("evictor", None)])
            calls, release, free = self.host(21.9)
            with release, free, self.assertLogs("vllm.qwen38fn_worker", "INFO") as logs:
                self.assertEqual(worker.initialize_from_config(self.arena(int(17.9 * gib))), "allocated")
            # The extra roots go first, so the MemFree reading follows every release.
            self.assertEqual(calls, ["/rt/one", "/rt/two", "/m"])
            self.assertEqual(logs.output[:2], [
                "INFO:vllm.qwen38fn_worker:host page cache released under /rt/one (7 files) before the KV arena",
                "INFO:vllm.qwen38fn_worker:host page cache released under /rt/two (7 files) before the KV arena"])
            # The model-root line keeps candidate 9's exact shape.
            self.assertIn("host page cache released under /m (7 files); MemFree 21.90 GiB, "
                          "KV arena 17.90 GiB + margin 2.00 GiB", logs.output[2])
        with mock.patch.dict(os.environ, {"QWEN38FN_HOST_RELEASE_ROOTS": "/rt/one:/rt/two"}):
            self.run_worker(body, cls="HostCacheReleaseWorker")

    def test_release_roots_default_to_none_and_reject_relative_or_root_paths(self):
        def body(worker, evictor_cls):
            self.assertEqual(self.module.RELEASE_ROOTS, ())
            parse = self.module.parse_release_roots
            self.assertEqual(parse(""), ())
            self.assertEqual(parse("/a:/b"), ("/a", "/b"))
            for bad in ("rel/path", "/", "/a::/b"):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    parse(bad)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("QWEN38FN_HOST_RELEASE_ROOTS", None)
            self.run_worker(body, cls="HostCacheReleaseWorker")

    def test_the_text_only_worker_is_unchanged(self):
        def body(worker, evictor_cls):
            with mock.patch.object(self.module, "release_tree") as release:
                worker.load_model()
                self.assertEqual(worker.initialize_from_config(self.arena(1)), "allocated")
            release.assert_not_called()
        self.run_worker(body)


if __name__ == "__main__":
    unittest.main()
