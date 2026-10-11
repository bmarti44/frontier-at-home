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
"""
from checkpoint_cache import CheckpointCacheEvictor
from glm53_worker import ReleaseLoadCacheWorker


class CheckpointCacheReleaseWorker(ReleaseLoadCacheWorker):
    def load_model(self, *args, **kwargs):
        self._checkpoint_evictor = CheckpointCacheEvictor(self.vllm_config.model_config.model)
        self._checkpoint_evictor.start()
        return super().load_model(*args, **kwargs)

    def compile_or_warm_up_model(self):
        try:
            return super().compile_or_warm_up_model()
        finally:
            self._checkpoint_evictor.stop()
