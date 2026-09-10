# Tiny native loading/parity preparation 003: FAIL

Selecting the pinned stock Torch fallback allowed all three tiny full forwards.
The partial-loader arm then failed at its first embedding dtype assertion. The
meta skeleton had been initialized in FP32, which native loading intentionally
prefers over its BF16 default. No forward-parity verdict was reached.

The bounded correction must construct the meta skeleton with native `_from_config`
and explicit BF16 initialization while retaining the stock strict-FP32 dtype plan.
It must verify every loaded tensor's shape, dtype and bytes against the full-load
arm, leave every unrelated parameter on meta, and retain the same per-layer and
full-logit assertions at lengths 5, 64 and 65. No checkpoint math, runtime files,
serving configuration or native reference data may change. All original evidence
and failed assertions remain archived; this is synthetic preparation only.
