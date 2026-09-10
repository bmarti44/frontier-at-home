"""Optional startup cleanup for explicit-budget GLM development launches."""
import gc
import os

import torch
from vllm.v1.worker.gpu_worker import Worker

_RELEASE_LOAD_CACHE = os.environ.get('GLM53_RELEASE_LOAD_CACHE', '0') == '1'


class WarmupCleanupWorker(Worker):
    def determine_available_memory(self):
        if _RELEASE_LOAD_CACHE:
            gc.collect()
            torch.accelerator.empty_cache()
        budget = super().determine_available_memory()
        # Match the native automatic-sizing path before allocating the real KV.
        # Live model tensors and the returned cache budget remain untouched.
        gc.collect()
        torch.accelerator.empty_cache()
        return budget
