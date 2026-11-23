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

# Patch 0003 check: vLLM's Triton prefill kernel (non-causal, varlen) against exact
# attention (fp32 SDPA per sequence) at the vision encoder's shape (16 heads x 64).
import torch.nn.functional as F
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
torch.manual_seed(0)
lens = [257, 1024, 2704]
cu = torch.tensor([0] + list(__import__("itertools").accumulate(lens)), dtype=torch.int32, device="cuda")
n, h, d = cu[-1].item(), 16, 64
q, k, v = (torch.randn(n, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
o = torch.empty_like(q)
context_attention_fwd(q, k, v, o, b_start_loc=cu[:-1], b_seq_len=cu[1:] - cu[:-1],
                      max_input_len=max(lens), is_causal=False, softmax_scale=d ** -0.5)
errs = []
for i in range(len(lens)):
    s0, s1 = cu[i].item(), cu[i + 1].item()
    ref = F.scaled_dot_product_attention(*(x[s0:s1].float().transpose(0, 1) for x in (q, k, v)),
                                         scale=d ** -0.5).transpose(0, 1)
    errs.append((o[s0:s1].float() - ref).abs().max().item())
out["vision_triton_attn_max_abs_err_vs_fp32"] = errs
out["vision_triton_attn_ok"] = all(e < 2e-2 for e in errs)
print(json.dumps(out))
