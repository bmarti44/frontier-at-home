"""Optional startup cleanup for explicit-budget GLM development launches."""
import gc

import torch
from vllm.v1.worker.gpu_worker import Worker


class WarmupCleanupWorker(Worker):
    def determine_available_memory(self):
        budget = super().determine_available_memory()
        # Match the native automatic-sizing path before allocating the real KV.
        # Live model tensors and the returned cache budget remain untouched.
        gc.collect()
        torch.accelerator.empty_cache()
        return budget
