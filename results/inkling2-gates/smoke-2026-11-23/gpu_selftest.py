#!/usr/bin/env python3
"""Runtime GPU self-test (DIAGNOSTIC): bf16 matmul, device cc, pageable-memory attribute,
and Inkling 2 kernel-path selection as the installed vLLM would make it."""
import json
import torch

out = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
       "cc": list(torch.cuda.get_device_capability(0))}
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
out["matmul_ok"] = bool(torch.isfinite(a @ a).all().item())
try:
    from cuda.bindings import driver as cu  # cuda-python, if present
    _, dev = cu.cuDeviceGet(0)
    attr = cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
    out["pageable_uses_host_page_tables"] = int(cu.cuDeviceGetAttribute(attr, dev)[1])
except Exception as e:  # noqa: BLE001
    import ctypes
    lib = ctypes.CDLL("libcuda.so.1")
    lib.cuInit(0)
    dev = ctypes.c_int()
    lib.cuDeviceGet(ctypes.byref(dev), 0)
    val = ctypes.c_int()
    rc = lib.cuDeviceGetAttribute(ctypes.byref(val), 100, dev)  # PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES
    out["pageable_uses_host_page_tables"] = val.value if rc == 0 else f"rc={rc} ({e!r})"
import vllm
out["vllm"] = vllm.__version__
from vllm.models.inkling2.nvidia.ops import use_sm100_kernels
out["inkling2_use_sm100_kernels"] = bool(use_sm100_kernels())
import flashinfer
out["flashinfer"] = flashinfer.__version__
print(json.dumps(out))
