# G5 + final production cells — Laguna S 2.1 — PASS (integration proven)

Window: 2026-08-22 (W5). This window used the REAL production switch path
end-to-end: `sudo 52_engine_switch.sh laguna` (hash + shared-library +
shard-sibling verification, transient unit on 8013 behind the auth chain,
readiness + semantic probe), suites against the switch-launched server,
then `sudo 52_engine_switch.sh qwen38-1m` back. Two switch-gate defects
were caught and fixed by the gates themselves along the way (per-slot
readiness constant from the 524K shape; semantic-probe 64-token budget
too small for max-thinking) — rollback restored qwen production cleanly
on every failed attempt.

## Tool-call probe (toolcall-v1, 20 pinned cases, greedy, same harness)

| model | passed | score |
|---|---|---|
| qwen38-1m (production) | 19/20 | **0.95** |
| laguna-s-2.1 (DFlash n4, thinking max) | 14/20 | 0.70 |

Laguna's misses cluster in enum-constrained arguments and
distractor-selection; qwen missed one case. On THIS OpenAI-tool-schema
harness the production default is markedly stronger — Laguna's published
agentic strength (Terminal-Bench 2.1 70.2) is not reproduced by this
probe format. Honest negative for the agentic-comparison hypothesis.

## Final strict speed cells (switch-launched production shape:
UD-Q4_K_XL, -c 393216 --parallel 4, DFlash n-max 4, thinking on)

| ctx | decode tok/s (median, 5 strict reps) |
|---|---|
| 0 | 25.55 |
| 28672 | **27.52** |

suite_valid = true; per-token SSE timestamps; Laguna output tokenizer
pinned (809240f7…). Reference: qwen38-1m production README cell 26.71
@28K. G2 code probes on this config: 28-45 tok/s (acceptance-dependent).

## Verdict

Integration + G5 PASS. Qualification bundle complete across
G1/G2/G3/G4/G5; Laguna S 2.1 qualifies as a code-focused switchable
engine. Per owner decision it is NOT the serving default (qwen38-1m
remains); switch in with `sudo scripts/52_engine_switch.sh laguna`.
