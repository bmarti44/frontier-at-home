# GLM-5.2 on the Spark — consolidated status & decision sheet (2026-07-26)

Single source of truth for the GLM-5.2 parity effort. Every number below is
measured, committed, and (where gated) sol-reviewed. Evidence index at the
bottom.

## The bar (DSV4 production baseline)

467 t/s prefill · 18.4 t/s decode · <2 s warm TTFT · 32K context.
Note: an earlier draft claimed DSV4's warm-TTFT figure was measured under
llama.cpp prefix-cache reuse — RETRACTED per the sol closing audit: the
committed 1.218 s record is a prefix-cache-cold tiny-prompt probe. A matched
DSV4-vs-GLM same-fixture head-to-head remains open.

## Final measured ledger

| metric | DSV4 | GLM-5.2 (best measured) | status |
|---|---|---|---|
| warm TTFT, exact replay | ~1.2 s (unmatched probe) | **1.76 s** | meets the absolute <2 s steady-state exact-replay threshold (byte-identical; matched head-to-head vs DSV4 open) |
| warm TTFT, multi-turn agent | <2 s | **5.6 s** (exact-prefix semantics, probe-gated) / ~150 s (strict default) | NOT met; ~2.2–2.7 s unvalidated projection after fix ladder below |
| context window | 32K | 32K allocated; **11.6K functional** + retrieval beyond row 8192 proven | partially validated (full-depth ~30K probe open) |
| decode | 18.4 t/s | **1.6–1.8 t/s** | NOT met — single-Spark physics (fully profiled) |
| prefill | 467 t/s | **~23 t/s** | NOT met — single-Spark physics |
| fidelity | — | **zero loss** on every adopted lever (byte chains + 100-case NLL 0.4515 / top-1 0.834 vs hosted) | MET |
| switching | — | one command both ways, 40+ verified round trips | MET |

Decode/prefill: every faithful lever is landed (pinned 68–72 GB expert cache,
parallel fetch ×6, batchall prefill, deterministic dispatch, kv-disk align)
or measured to its ceiling (MTP +10%, REAP rejected, compression dead,
kernel read path ~55 GB/s). Remaining faithful levers converge at ~3–4 t/s
decode. Parity requires: second Spark (NVMe-oF miss tier), a measured
fidelity trade, or DSV4 remaining prod.

## Multi-turn TTFT: the 5.675 s turn, attributed to the millisecond

evict re-store (920 MiB to a NEW file every turn) 1227 ms · shard load
295 ms · 22-token suffix prefill @8.84 t/s 2487 ms · first-token decode
1001 ms · session/HTTP ~660 ms.

Fix ladder (bounded engine work, in order): (1) stop the redundant per-turn
re-store −1.2 s; (2) live-rewind to the common prefix the server already
computes instead of evict+load −0.3 s; (3) small-suffix prefill optimization
−1.5–2 s. Projected floor ≈ 2.2–2.7 s — an UNVALIDATED projection
(additivity not demonstrated; per sol closing audit). Rung 4 is the decode
floor itself. Component evidence: logs/loadprof1/appended-turn-components.txt.

## The resume-"bug" saga — demonstrated mechanism, one invariant open

The dominant demonstrated mechanism is deep-layer numeric path-dependence on
evaluation chunk shape (L0 rows byte-exact across everything; L40 rows
differ between any two chunkings), amplified by greedy decoding — the same
equivalence class as llama.cpp prefix-cache reuse. Token ids at the junction
are identical (TOKDUMP); restore is content-faithful where testable. The
v4.8d guard (in the tree binary) enforces byte-canonical-or-cold — stricter
than industry practice. HOWEVER, per the sol closing audit: corruption is
NOT excluded while the L40 same-lineage store→load round-trip violation
stands unexplained (a pure byte-copy should preserve those rows). The strict
guard REMAINS the recommended serving posture until the targeted
no-eval-between round-trip dump passes — which also weakens the case for
relaxing the guard default in decision 1 below.

## Decisions on the table (owner: Brian)

1. **Resume-semantics default**: relax the guard to exact-prefix resume
   (llama.cpp-equivalent; enables the 5.6 s → 2.3–2.9 s ladder) or keep
   strict byte-canonical (appends ~150 s cold). Two-line polarity change +
   validation window + sol review. (Attempted flip was blocked by the
   permission layer as a policy change — correctly.)
2. **Decode/prefill parity route**: second Spark over ConnectX-7 NVMe-oF
   (12–24 GB/s miss tier) · accepted measured fidelity trade · DSV4 stays
   prod with GLM one command away.
3. **Upstream filing**: docs/ds4-glm-resume-frontier-bug-2026-07-26.md is
   ready, recast as the L40 round-trip question + resume-semantics
   documentation note. Public report — needs go-ahead.

## Evidence index

results/glm52-gates/loadprof-2026-07-25.json (master findings ledger,
20+ sections) · G4-bench.json (perf snapshots incl. REAP three-way) ·
logs/loadprof1/ (raw evidence + sol trails) · harness/ (all probes,
reproducible) · gate JSONs G0–G5 with sol PASS trails · engine patch
harness/ds4-iq2xxs-down-cuda.patch (v4.8d) + dsv4 tree commits through
v4.8e diagnostics. Safety: ~40 GLM windows this arc, zero freezes, zero
kill-floor events, DSV4 restored and VERIFY OK after every window.
