# Vision workspace source audit

Read-only inspection of pinned build-source-005. No GPU vision run or measured
memory claim follows. This records the next bounded workspace question while
indexer candidate 2 is being audited.

The selected vision configuration is 24 layers, hidden size 1,024, 16 heads,
intermediate size 4,096, projection intermediate size 10,240, patch size 14,
temporal patch size 2 and spatial merge size 2. The actual values come from
processor-metadata-002/config.json, not constructor defaults or source comments.

The checkpoint processor config contains 8,000 image tokens and 240,000 video
tokens. The actual Glm5NextProcessor.from_pretrained path caps video to 30,000
(_MAX_VIDEO_TOKENS), while preserving the checkpoint image budget. With a
16-frame policy, use the resolved processor budget and aligned canvas; frame
count alone does not bound vision memory. Each merged vision token represents
four pre-merge patches. An 8,000-token image therefore permits 32,000 patches,
and a 30,000-token video permits 120,000 patches before alignment effects.

Do not multiply those maxima blindly by all four request slots.
compute_mm_encoder_budget takes the maximum of scheduler encoder budget and
largest enabled item, while MultiModalBudget separately limits encoder items
per batch. The scheduler initializes its encoder compute/cache budgets from
max_num_batched_tokens (2,048 here). Resolve the actual processor's maximum
item counts, final budget and worker batch splitting before constructing a
worst-case component fixture. Four allowed images can require more than one
encoder batch; this is independent of the four language-model context slots.

The vision forward path includes patch embedding, QKV projection, q/k RMSNorm,
contiguous rearrangement, concatenated q/k rotary embedding, encoder attention,
MLP gate/up and down projections, spatial merging and final projection. Several
large tensors can coexist. rot_pos_emb builds CPU position IDs and transfers
them with non_blocking=True, which does not establish pinned memory by itself.
A probe must preserve actual behavior and either retain pinned metadata through
the supported encoder_metadata route or measure and justify that transfer.
Do not change the previously approved processor/sampler based on this audit.

Sources in the pinned vLLM tree:
- models/glm5next/nvidia/multimodal.py:188, 450, 480, 551, 635.
- transformers_utils/processors/glm5next.py:50, 301, 798.
- v1/core/encoder_cache_manager.py:282.
- multimodal/encoder_budget.py:49.
- config/scheduler.py:238.
