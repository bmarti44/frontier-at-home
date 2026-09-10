"""Optional startup cleanup for explicit-budget GLM development launches."""
import gc
import json
import os

import torch
from vllm.v1.worker.gpu_worker import Worker

_RELEASE_LOAD_CACHE = os.environ.get('GLM53_RELEASE_LOAD_CACHE', '0') == '1'
_RELEASE_WARMUP_CACHE = os.environ.get('GLM53_RELEASE_WARMUP_CACHE', '0') == '1'


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

    if _RELEASE_WARMUP_CACHE:
        def compile_or_warm_up_model(self):
            result = super().compile_or_warm_up_model()
            before = torch.cuda.memory_reserved()
            gc.collect()
            torch.accelerator.empty_cache()
            after = torch.cuda.memory_reserved()
            print(json.dumps({'event': 'glm53_release_warmup_cache',
                'reserved_before_bytes': before, 'reserved_after_bytes': after}), flush=True)
            return result
