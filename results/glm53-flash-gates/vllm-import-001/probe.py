import importlib
import json
import os
import torch
import vllm
from pathlib import Path

root = Path(vllm.__file__).parent
loaded = []
for name in ('_C_stable_libtorch.abi3.so', '_moe_C_stable_libtorch.abi3.so'):
    torch.ops.load_library(str(root / name))
    loaded.append(name)
rust = importlib.import_module('vllm._rust_tool_parser')
print(json.dumps({'qualification': 'native_import_only', 'verdict': 'PASS', 'torch_version': torch.__version__,
                  'vllm_version': vllm.__version__, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                  'cuda_initialized': torch.cuda.is_initialized(), 'native_libraries': loaded,
                  'rust_extension': rust.__file__, 'model_loaded': False, 'kernel_execution': False}))
