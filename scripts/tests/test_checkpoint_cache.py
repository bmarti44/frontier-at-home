"""checkpoint_cache.CheckpointCacheEvictor: evict a checkpoint's page cache during engine startup."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import threading
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


if __name__ == "__main__":
    unittest.main()
