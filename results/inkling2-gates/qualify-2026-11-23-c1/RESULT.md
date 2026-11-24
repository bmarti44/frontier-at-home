# Candidate 1: superseded before it ran (no verdict)

Candidate 1 was frozen at `382e7a8c` and never launched. No weights were loaded for it, and production was not stopped for it.

**Why.** After the freeze, its scorer was run against the screen carrier's real evidence (`../tune-screen-2026-11-23/a6`).
- Clause 11 required `"K = max tokens per event"` in `metadata["method"]` of `speed/speed.json`.
- The block-aware bench (`scripts/30_bench_speed_blocks.py`) writes no `method` key. It writes `metadata["decode_method"] = "block-aware: (client_completion_tokens - max_tokens_per_event) / (t_last_event - t_first_event)"`.
- So every real run would have scored `NO_RESULT`. The 73 mutations passed only because the synthetic bundle used the same invented key.

Reviewer A found it independently (A-M3c1-H1, high). Reviewer B found no high or critical run-safety issue and added two lows (B7 soak abort, B8 abort scoring).

Candidate 2 (`../qualify-2026-11-23-c2`) carries the fix, two real-evidence checks in the mutation suite, and the B7/B8 abort text.
