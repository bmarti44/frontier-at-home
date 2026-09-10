# One native BF16 layer: bounded memory falsifier

This is a real-weight, synthetic-activation feasibility probe. It cannot supply
native full-model reference probabilities, fidelity, serving speed or context
qualification. Preserve every failure and stop at the first failed prerequisite.
No serving profile, installed library, model math or default changes.

Use layer 8, one of the largest forward KDA layers, because its 14,825,277,272
stored tensor bytes occupy only three complete shards (115–117) of pinned release
`zai-org/GLM-5.3-Flash-BF16` at `a5b45eb41df6402735dedc900be14a42e8d5e538`.
Hold at most one complete downloaded shard buffer; verify its complete LFS SHA256
before using any tensor. Validate its header against the pinned metadata. Copy
only the declared layer tensors, release each shard, and use the stock checkpoint
conversion and strict FP32 dtype plan on a BF16-initialized meta skeleton. Force
sequential HF materialization. Every other model parameter must remain on meta.

Select the stock Torch fallback, eager attention/experts, eval and inference_mode,
as in the independently reproduced tiny test. Use synthetic BF16 HC activations
of shape [1,516,4,4096], chosen from the maximum serialized Kimi prompt length.
This does not reconstruct the actual layer-8 activations of those prompts.
Bind the synthetic generator and later verified public seed. Persist pinned
weight and activation staging through H2D completion; do not use pageable H2D.
Require finite, complete output with the expected shape/dtype and retained bytes.

Use the existing inference lock, hardened wrapper, identity guard and host scorer.
Require stable start memory >=110 GiB, a 40 GiB kill floor, fresh containment with
MemoryHigh=62 GiB, MemoryMax=64 GiB, MemorySwapMax=0, OOMPolicy=kill,
KillMode=control-group and a 600-second timeout. The wrapper must independently
verify 64+40 GiB fits measured available memory. Approximate worst conversion
residency is 35 GB; three copies of a converted layer are about 44.5 GB, plus a
10 GB workspace reserve and 2 GiB process allowance, below 64 GiB. These are
conservative admission estimates, not measured peaks. The native FP32 plan alters
stored byte counts. At 516 tokens KDA pads to 576; a single decay mask is
1,207,959,552 bytes and expression temporaries add to it. Measure actual residency.

Before the run, freeze clean source, runtime inventory, metadata, tokenizer/corpus,
scorers, exact arguments and environment. Obtain verified public randomness later.
Require no observed change in broad pre-freeze global swap-in/out or used swap.
Hash/shape/dtype mismatch, unexpected materialization, short output, nonfinite
values, timeout, OOM/Xid, swap, cgroup pressure, insufficient memory, missing
identity or surviving descendants yields FAIL. The unchanged host scorer must
pass along with every inner check; missing evidence is never a PASS. Retain raw
commands, pinned-byte/copy receipts, output bytes, timestamped host observations
and terminal cleanup. Apply malformed-evidence controls to the new loader.

A PASS authorizes only consideration of the next reference preparation step.
It does not establish end-to-end native equivalence, full reference feasibility,
a 100-case fidelity result or production promotion.
