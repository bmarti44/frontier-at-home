import importlib
import json
import multiprocessing
import os
from pathlib import Path
import sys


def inspect():
    import torch
    import exllamav3_ext
    import vllm_exl3_c
    import vllm_exl3.exl3
    import vllm
    prefix = Path(sys.prefix)
    if any(not Path(p).is_relative_to(prefix) for p in sys.path):
        raise ValueError('packaged sys.path escapes new prefix')
    if not sys.flags.isolated or not sys.dont_write_bytecode or torch.cuda.is_initialized():
        raise ValueError('isolated CPU import flags or CUDA state mismatch')
    if not callable(exllamav3_ext.exl3_moe) or getattr(vllm_exl3_c, 'P2B_MOE_ABI_VERSION', 0) < 2:
        raise ValueError('native MoE ABI unavailable')
    return {'prefix': sys.prefix, 'executable': sys.executable, 'sys_path': sys.path,
            'isolated': sys.flags.isolated, 'dont_write_bytecode': sys.dont_write_bytecode,
            'cuda_initialized': torch.cuda.is_initialized(), 'vllm': vllm.__version__,
            'native_moe_abi': vllm_exl3_c.P2B_MOE_ABI_VERSION}


def child(queue):
    queue.put(inspect())


if __name__ == '__main__':
    parent = inspect()
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    result = queue.get(timeout=60)
    process.join(timeout=30)
    if process.exitcode != 0 or parent != result:
        raise ValueError('packaged spawn child differs')
    print(json.dumps({'verdict': 'PASS', 'qualification': 'packaged_native_CPU_import_and_spawn_only',
                      'parent': parent, 'child': result, 'model_loaded': False, 'kernel_execution': False}))
