# Long-context crash: bounded source triage

This is CPU/source analysis of context-direct-003, not a model run, kernel
correctness result or context-capacity result. Crash attribution remains
**NO_RESULT**. Server 017 remains available and no runtime file was changed.

`analyze.py` extracts the actual failed scheduler batch from the preserved log:
four 32-token prefills, prior lengths 52992/52928/52896/52864, and no generated
tokens. For the convolution branch, the rebased positions are
0/32/64/96/128, giving 16 temporal tiles and 96 channel tiles (1,536 programs)
over 24,576 merged channels. QKV may retain the fused projection's 24,896-element
token stride; compact materialization would use 24,576. Both cases and the
convolution's history/tail reads are included in the arithmetic.
The GLM call does not supply the optional prefix-cache history offsets. The
local input/state-row address arithmetic therefore does not grow directly
with the absolute 529xx history. Actual GPU state IDs, metadata corruption,
data-dependent failures and earlier layers remain untested.

The asynchronous `.tolist()` error occurs before the current KDA layer's chunk
kernels. It cannot identify the kernel that first accessed bad memory. The
kernel journal's Xid31 virtual-read fault is a real failure, preserved in the
original attempt.

## Relevant upstream reports

[vLLM issue 54317](https://github.com/vllm-project/vllm/issues/54317) reports
GLM-5.3 failures on B200 surfacing in several kernels, including the same KDA
synchronization. The report's mixed-batch discussion does not describe this
pure-prefill batch. Its disabled-PDL attempt still crashed. These observations
do not establish that the Spark has the same defect or justify adopting a
workaround without reproduction.

[vLLM issue 49896](https://github.com/vllm-project/vllm/issues/49896) reports
nonfinite indexer scores producing invalid selected indices for DeepSeek-V4
on SM120. This suggests testing nonfinite inputs to the shared top-k code.
Our GLM attention conversion has different bounds checks, so the reported
DeepSeek crash chain cannot be assumed to apply here.

`check_rank_comparisons.py` checks only the final insertion-phase comparison:
four finite values, including ties, produce four distinct ranks; four NaNs
produce rank zero four times. This is a conditional numerical observation.
Neither nonfinite native logits nor their admission through earlier histogram
stages has been established. No production change is justified by this alone.

The first convolution arithmetic draft omitted the wider input stride,
channel-grid factor and extra history reads. It is preserved under `initial/`
with the review findings; the revised calculation retains a NO_RESULT verdict
for CUDA crash attribution.

The next useful device checks are small tensors with synchronization between
convolution, state gathering and recurrence, and between index compression,
top-k selection and attention gathering. They must run after the serving model
is safely stopped; no second model or GPU test was run during this analysis.
