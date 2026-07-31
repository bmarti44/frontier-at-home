# Historical telemetry aggregate notice

`w1-telemetry-loaded-probes-1701ba8.json` is retained unchanged as a record of
the original investigation. It is informational and non-authoritative because
it contains hand-authored aggregates rather than hash-bound raw logs and was
recorded before the lifecycle-precision candidate was frozen.

The authoritative replacement is
`w1-telemetry-probe-893f637-post-freeze-1/`. Its manifest binds the frozen
candidate and artifacts, its `raw.jsonl` preserves every lifecycle and memory
sample plus the direct-I/O witnesses, and the fixed repository scorer derives
its `summary.json`.
