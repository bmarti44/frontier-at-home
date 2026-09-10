# Reuse the observed decode tail seed cache

Attempt004 generated eight `_kpool_tail_seed_kernel` files at first decode.
This is a prepared-cache coverage failure, preserved in the committed direct
attempt. The fixed regression calls the actual existing kernel preparation
function and requires those eight captured files to be copied, byte-identical
except for relocated Triton group paths. All group children must exist below
the new state directory. It does not launch a model or compile CUDA.

The smallest correction is to add the now-preserved server019 state to the
existing ordered preparation sources. No kernel, scorer, model or request prompt
changes are part of this correction. Existing completed profile code reviews
remain scoped to their frozen candidates. The next run needs a new cache freeze.
