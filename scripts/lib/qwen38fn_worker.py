"""vLLM worker for Qwen3.8-Flash-Next: keep checkpoint page cache out of startup.

The model loads 123.6 GiB of checkpoint twice: target weights, then the MTP
drafter's scan of every shard. On unified memory, the read-once page cache
left behind makes the next large GPU allocation (drafter weights, the 17.9
GiB KV arena) meet MemFree of 1-3 GiB, and the driver logs ``NVRM ...
NV_ERR_NO_MEMORY``. Evicting the checkpoint's unmapped pages from before the
weights load until warmup ends kept MemFree at 12.5 GiB or more, with no NVRM
lines (results/qwen38fn-gates/diag-drafter-2026-10-11, arm C). Nothing runs
while serving. Keeps the GLM-5.3 allocator release. Selected with
``--worker-cls qwen38fn_worker.CheckpointCacheReleaseWorker`` (``PYTHONPATH``
must include scripts/lib).

``HostCacheReleaseWorker`` also covers the page cache other engines leave
behind. Candidate 7 passed the MemAvailable admission with qwen38-1m's GGUF
still cached, met the 17.9 GiB KV arena at MemFree 10.2 GiB, and logged 38
NVRM lines (results/qwen38fn-gates/qualify-2026-10-11-c7). This worker drops
the cache of every file under the model root before the weights load and
again before the arena. It then refuses the arena unless MemFree covers it
plus ``KV_FREE_MARGIN_GIB``, so startup fails cleanly instead. Both steps run
once at startup. Selected with ``--worker-cls
qwen38fn_worker.HostCacheReleaseWorker``.
"""
import logging
import os

from checkpoint_cache import CheckpointCacheEvictor, mem_free_gib, release_tree
from glm53_worker import ReleaseLoadCacheWorker

logger = logging.getLogger("vllm.qwen38fn_worker")
KV_FREE_MARGIN_GIB = 2.0


class CheckpointCacheReleaseWorker(ReleaseLoadCacheWorker):
    def load_model(self, *args, **kwargs):
        self._checkpoint_evictor = CheckpointCacheEvictor(self.vllm_config.model_config.model)
        self._checkpoint_evictor.start()
        return super().load_model(*args, **kwargs)

    def compile_or_warm_up_model(self):
        try:
            return super().compile_or_warm_up_model()
        finally:
            evictor = getattr(self, "_checkpoint_evictor", None)
            if evictor is not None:
                evictor.stop()


class HostCacheReleaseWorker(CheckpointCacheReleaseWorker):
    def load_model(self, *args, **kwargs):
        # The model root holds every engine's model files on this host.
        self._model_root = os.path.dirname(os.path.abspath(self.vllm_config.model_config.model))
        release_tree(self._model_root)
        return super().load_model(*args, **kwargs)

    def initialize_from_config(self, kv_cache_config):
        root = getattr(self, "_model_root", None) or os.path.dirname(
            os.path.abspath(self.vllm_config.model_config.model))
        files = release_tree(root)
        # Every KVCacheTensor views one shared backing allocation of ``size``
        # bytes (vllm/v1/worker/utils.py asserts a single size and allocates it once).
        arena = max([t.size for t in kv_cache_config.kv_cache_tensors]
                    + [self.cache_config.kv_cache_memory_bytes or 0]) / 2**30
        free = mem_free_gib()
        message = (f"host page cache released under {root} ({files} files); MemFree {free:.2f} GiB, "
                   f"KV arena {arena:.2f} GiB + margin {KV_FREE_MARGIN_GIB:.2f} GiB")
        if free < arena + KV_FREE_MARGIN_GIB:
            logger.error("%s: refusing to allocate the KV arena", message)
            raise RuntimeError(f"{message}: refusing to allocate the KV arena; page cache outside "
                               f"{root} would leave the driver short of free memory (NVRM)")
        logger.info(message)
        return super().initialize_from_config(kv_cache_config)
