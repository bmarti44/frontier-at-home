"""CPU-only native TorchDynamo switch control; no GPU or compiler backend."""
import json
import os
import torch

calls = []
def backend(graph, inputs):
    calls.append(True)
    raise RuntimeError("control backend was reached")

@torch.compile(backend=backend)
def leaf(value):
    return value + 1

result = {"torch_version": torch.__version__, "disabled": os.environ["TORCH_COMPILE_DISABLE"],
          "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "device": "cpu"}
try:
    value = leaf(torch.tensor([1, 2, 3], device="cpu"))
    result["output"] = value.tolist()
except Exception as error:
    result["failure"] = str(error)
result["backend_calls"] = len(calls)
if result["disabled"] == "1":
    passed = result.get("output") == [2, 3, 4] and result["backend_calls"] == 0
else:
    passed = "control backend was reached" in result.get("failure", "") and result["backend_calls"] == 1
result["verdict"] = "PASS" if passed else "FAIL"
result["qualification"] = "cpu_dynamo_switch_only"
print(json.dumps(result))
raise SystemExit(0 if passed else 1)
