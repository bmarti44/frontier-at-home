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

``release_tree`` applies the same advice once to every file under a model
root. Another engine's model files stay cached after that engine stops.
MemAvailable counts them as free, but the driver cannot use them for a large
allocation. Candidate 7 met its KV arena at MemFree 10.2 GiB with qwen38-1m's
GGUF still cached (results/qwen38fn-gates/qualify-2026-10-11-c7).
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


def release_tree(root: str) -> int:
    """Drop the unmapped page cache of every regular file under ``root``.

    Symlinks are neither followed nor advised, so nothing outside ``root`` is
    touched. A file this user cannot open, or one deleted during the walk, is
    skipped. Any other error is raised. Returns the number of files advised.
    """
    if not root or os.path.realpath(root) == "/" or not os.path.isdir(root):
        raise ValueError(f"refusing to release page cache under {root!r}")
    count = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except (PermissionError, FileNotFoundError):
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            count += 1
    return count


def mem_free_gib(meminfo: str = "/proc/meminfo") -> float:
    with open(meminfo) as stream:
        for line in stream:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) / 2**20
    raise ValueError(f"no MemFree line in {meminfo}")
