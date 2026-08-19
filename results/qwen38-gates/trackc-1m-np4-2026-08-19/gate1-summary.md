# Track C gate 1 — 4x262K native slots (-c 1048576 -np 4), q8_0 KV — 2026-08-19

Config: Q4_K_M, b10488, -c 1048576 --parallel 4 (four native-262K slots,
NO YaRN — 4x262144 = 1048576 exactly), -ctk/-ctv q8_0, MTP n8 p0.6,
--cache-reuse 256, capped unit MemoryHigh=78G/Max=88G.

PASS on all gate-1 checks:
- Startup + health at full 1M cell allocation; unit resident 19 GiB at
  ready (q8_0 KV commits lazily), 26 GiB after the 28K bench. Worst-case
  full 4-slot fill ~+34 GB KV stays far inside the cap.
- **MTP works with -np 4** (draft_n 91, accepted 57 on the sanity gen);
  single-active code decode 28.79 tok/s — fastest measured config yet
  (q8_0 KV also halves cache bandwidth per token).
- Strict cells (bench-np4.json, suite valid): decode 17.44 @0 /
  23.06 @28K, prefill 694.8 @28K, TTFT 0.36 s short — parity with the
  qualified 32K profile.

Gate 2 (fidelity: deep needle retrieval + full accuracy suites under
q8_0 KV, ledger namespace trackc-np4q8) follows before any production
adoption, per the DSV4 quantized-KV rule.
