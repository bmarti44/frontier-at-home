# DIAGNOSTIC arm B only (diag-drafter-2026-10-11): reached through a scratch
# PYTHONPATH, never the production scripts/lib. Arm A showed the start-1
# allocation inside the drafter's DeepGEMM UE8M0 requant
# (fp8_utils.requant_weight_ue8m0_inplace). This arm takes the drafter's FP8
# MoE off DeepGEMM so that step never runs. The target's NVFP4 experts do
# not use DeepGEMM. Same stack-dump hook as arm A.
import faulthandler
import os
import signal
import sys

os.environ["VLLM_MOE_USE_DEEP_GEMM"] = "0"
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
