# Reference byte binding: PASS

The local tokenizer digest matches the pinned public suite. A bounded streaming
comparison verified every byte of the 1,268,776,960-byte BF16 output head and
8,192-byte final normalization vector against the pinned public artifacts.
Their complete remote safetensors SHA256 digests also match the previously
inventoried reference metadata. No model was stopped and no GPU work was done.

This establishes tokenizer/head/norm identity only. It does not establish that
public hidden-state replay equals native reference logits, qualify the current
EXL3/FP8 model, or supply missing candidate captures. The unchanged 100-case
quality gate remains open. The source and full comparison receipts are retained.
