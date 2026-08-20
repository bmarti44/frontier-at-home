# Track C gate 4 — full-load residency proof + recovery drill — 2026-08-19

Same launch as the qwen38-1m profile (f16 KV, -np 4, MTP n8p6), caps
88G/95G, memory sampled every 10 s (memory-samples.log).

## Full fill — 4 slots to ~260K tokens each (slotfill.json)
| slot | prompt_n | retrieval | prefill t/s |
|---|---|---|---|
| 0 | 260,361 | found | 421.8 |
| 1 | 260,334 | found | 422.0 |
| 2 | 260,363 | found | 421.6 |
| 3 | 260,354 | found | 421.4 |

Total resident context: **1,041,412 tokens**; 4/4 needle retrievals
correct at 99.3% of the per-slot cap.

## Memory findings
- **cgroup MemoryCurrent is blind to the CUDA unified-memory KV**: unit
  peak read 22 GiB while system MemAvailable fell to **11 GiB minimum**
  (~66 GB KV committed, matching the 64 KiB/token f16 math). The
  cgroup caps are therefore only partial containment on GB10; the
  system-level memwatch is the real guard.
- Consequence: an 18 GiB watchdog floor would false-trip near full
  load. The qwen38-1m profile adopts the **8 GiB floor** (the same
  owner-accepted floor DSV4's 1M profile runs); the 32K profile keeps
  18 GiB.

## Recovery drill
SIGKILL to the fully-loaded unit: unit gone in <5 s, zero listeners
left on the port, production restored and verified via the standard
trap. Contained failure, clean recovery.
