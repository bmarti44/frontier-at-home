# Dev/diagnostic tooling from the 2026-07-23/24 speed-tuning sessions

Companion tooling for docs/speed-tuning-2026-07-23.md and
docs/speed-paths-2026-07-24.md. All scripts target a loopback llama-server
(default 127.0.0.1:8011; the dev-bench copies used :8021) and are
DEV-ONLY — none of them respect the production launcher's safety gates.

- `gguf-tensors.py FILE.gguf` — parse a GGUF header without any GGUF
  library: tensor names, block range, MTP/NextN presence. Use it (or a
  512 KB HTTP range fetch of the header) BEFORE downloading 70+ GiB weights.
- `dsv4-bench.py [label]` — four-workload speed benchmark (prose, code,
  19K prefill+gen, 19K-ctx code) reading llama-server response timings
  (cache_n / prompt_n / prefill and decode tok/s).
- `dsv4-ngram-test.py` — context-echo workloads (code edit, repetitive
  JSON tool-calls) that exercise ngram speculative decoding; reports
  draft_n / draft_n_accepted.
- `dsv4-cache-test.py` — deterministic prefix-cache verification
  (cold / identical / shared-prefix requests).
- `dsv4-slot-test.py` — slot save/restore round-trip check (currently
  demonstrates the restore no-op documented as Bug A).
- `ds4-bench.py` — same workloads against the ds4 engine (:8012),
  wall-clock based since ds4 exposes no timing fields.
- See also docs/patches/reap-two-request-probe.sh (two-request crash probe)
  and docs/patches/mmid-duplicate-expert-ids.patch (the ported+extended fix,
  verified against llama.cpp-reapfix worktree @ 0dc74e33 base).
