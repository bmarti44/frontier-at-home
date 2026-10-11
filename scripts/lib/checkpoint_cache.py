"""Keep a checkpoint's page cache out of a large engine's startup (unified memory).

On GB10 the GPU allocates from host memory. During a load, the checkpoint's
read-once pages fill the page cache. The next large GPU allocation then
meets MemFree of 1-3 GiB, and the driver logs
``NVRM ... NV_ERR_NO_MEMORY`` while the kernel reclaims. That happened in
Qwen3.8-Flash-Next Phase 2 (results/qwen38fn-gates/diag-drafter-2026-10-11).

``posix_fadvise(POSIX_FADV_DONTNEED)`` drops those pages without privilege.
It skips pages a process has mapped, so in-place mappings (the PLE table) and
the loader's live mmap stay intact. The evictor runs one pass at start, one
pass every ``interval_s`` on a daemon thread, and one final pass at stop. A
pass that fails is raised by ``stop()``, never swallowed.
"""
from __future__ import annotations

import glob
import os
import threading

THREAD_NAME = "checkpoint-cache-evictor"


class CheckpointCacheEvictor:
    def __init__(self, model_dir: str, interval_s: float = 1.0) -> None:
        self.paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not self.paths:
            raise ValueError(f"no *.safetensors files in {model_dir}")
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def evict_once(self) -> None:
        for path in self.paths:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)

    def _run(self) -> None:
        try:
            while not self._stop.wait(self.interval_s):
                self.evict_once()
        except BaseException as error:  # surfaced by stop()
            self._error = error

    def start(self) -> None:
        self.evict_once()
        self._thread = threading.Thread(target=self._run, name=THREAD_NAME, daemon=True)
        self._thread.start()

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        if self._error is not None:
            raise RuntimeError("checkpoint page-cache eviction failed") from self._error
        self.evict_once()
