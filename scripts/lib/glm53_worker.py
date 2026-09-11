"""vLLM worker for GLM-5.3-Flash: release load-time allocator reserve.

The EXL3 instanttensor load leaves roughly 20 GiB of freed-but-cached blocks in
the torch caching allocator. With ``--kv-cache-memory-bytes`` vLLM skips memory
profiling and never empties that cache, so on unified memory the host sits ~20
GiB lower than the resident weights + KV imply (measured: MemAvailable 10 GiB
at idle instead of ~30 GiB). Emptying the cache before and after the (skipped)
profiling step returns it. Selected with ``--worker-cls
glm53_worker.ReleaseLoadCacheWorker`` (``PYTHONPATH`` must include scripts/lib).
"""
import gc

import torch
from vllm.v1.worker.gpu_worker import Worker


class ReleaseLoadCacheWorker(Worker):
    def determine_available_memory(self):
        gc.collect()
        torch.accelerator.empty_cache()
        budget = super().determine_available_memory()
        gc.collect()
        torch.accelerator.empty_cache()
        return budget
