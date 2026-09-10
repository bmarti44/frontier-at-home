# Tiny stock Torch loading/parity preparation 004: PASS

A fresh independent CPU-only reproduction matched all 310 tensor comparisons and
seven partial stages. The synthetic model has five layers, 12 experts and three
checkpoint shards; prompts have 5, 64 and 65 tokens. Stock checkpoint conversion
preserved all declared BF16/FP32 parameter bytes. The partial module retained only
its selected stage on CPU, with unrelated parameters remaining on meta. Every
layer output, returned non-null top-k tensor and final logit tensor matched the
full-load arm exactly. Seven mismatch/nonfinite/missing-stage controls rejected.

This does not measure whole-process streaming memory: full baseline and synthetic
source tensors remain resident, and selected weights are cloned from an in-memory
dictionary. No real model weights, streamed shard I/O, GPU execution, native
fidelity, context or speed were tested. Both arms explicitly selected the pinned
stock Torch fallback; FLA/Triton equivalence and recurrent/update execution remain
untested. Previous preparation failures 001–003 are preserved. No serving code or
installed runtime changed.
