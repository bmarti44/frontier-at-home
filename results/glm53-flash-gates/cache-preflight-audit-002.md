# Allocator preregistration refinement

The follow-up source audit includes manager padding, shared pool groups and the
null block. The expected values are now fixed in
`configs/decision-specs/glm53-cache-preflight.json`, before native execution.

KDA state is4,341,760bytes. Platform normalization first uses the provisional
MLA head512, rounds to128tokens, then SM121 index alignment256, giving8704tokens
per manager block. Actual packed MLA storage is656bytes/token, not512.
Each full262144-token request needs31attention blocks, four Mamba group blocks
and one tail block. Four requests plus the null block require145pool IDs.
The single backing allocation is`145*11*8704*(656+132/4)=9,565,304,320bytes`.
This remains a source-derived expectation until the actual allocator and manager
complete the preregistered checks.

Use actual shape/dtype helpers; default SD convolution state is`(3,24576)`BF16
and recurrent state`(64,128,128)`FP32. With APC off, Mamba's semantic block size
is262144. Its provisional padding`8704*512` is replaced by grouping with the
actual MLA page. Index storage pages are256original tokens (64pooled entries).
Expected groups are one combined MLA/index, one tail and four KDA groups with
9,9,8,8layers. Sum unique storage once.

The pinned processor factory keeps8000tokens/image and clamps video to30000.
Four images/request across four slots admit128000merged tokens; four videos
admit120000. Exact maximum-budget synthetic geometries are2240x2800 per image
and16frames1400x2100 per video. Maximum retained BF16 encoder outputs at width4096
are1,048,576,000bytes (1000MiB). This is not simultaneous scratch or total vision
memory: use actual encoder scheduling, pixel buffers and retained outputs.

Path correction to audit001: GLM model files are under
`vllm/vllm/models/glm5next/nvidia/`, not `model_executor/models/`.
Source inspection and integer/hash operations only; no Torch import or GPU work
occurred in the review. The root runner must confirm every expected value with
the pinned installed engine and preserve any mismatch as a failed attempt.
