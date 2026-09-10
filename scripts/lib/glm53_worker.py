"""Optional startup cleanup for explicit-budget GLM development launches."""
import gc
import json
import os

import torch
from vllm.v1.worker.gpu_worker import Worker

_RELEASE_LOAD_CACHE = os.environ.get('GLM53_RELEASE_LOAD_CACHE', '0') == '1'
_RELEASE_WARMUP_CACHE = os.environ.get('GLM53_RELEASE_WARMUP_CACHE', '0') == '1'
_TRIM_STARTUP_HEAP = os.environ.get('GLM53_TRIM_STARTUP_HEAP', '0') == '1'

if _TRIM_STARTUP_HEAP:
    import ctypes
    import time

    _malloc_trim = ctypes.CDLL(None).malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int

    def _startup_memory():
        values = {}
        with open('/proc/self/status') as stream:
            for line in stream:
                name, _, value = line.partition(':')
                if name in ('VmRSS', 'RssAnon'):
                    values[name] = int(value.split()[0])
        return {'rss_kib': values['VmRSS'], 'anonymous_rss_kib': values['RssAnon']}

    def _trim_startup_heap(phase):
        start = time.time()
        before = _startup_memory()
        result = _malloc_trim(0)
        after = _startup_memory()
        print(json.dumps({'event': 'glm53_trim_startup_heap', 'phase': phase,
            'pid': os.getpid(), 'start_unix': start, 'end_unix': time.time(),
            'trim_result': result, 'before': before, 'after': after}), flush=True)


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

    if _TRIM_STARTUP_HEAP:
        def determine_available_memory(self):
            if _RELEASE_LOAD_CACHE:
                gc.collect()
                torch.accelerator.empty_cache()
                _trim_startup_heap('before-profile')
            budget = super().determine_available_memory()
            gc.collect()
            torch.accelerator.empty_cache()
            _trim_startup_heap('after-profile')
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
