# MiMo-V2.6-Flash M3 acceptance: candidate 3 PASS

**Verdict: `PASS`.** The frozen scorer (`gate_verdict.py` at the freeze
`2a75f1fe`) returned `PASS` with no failed clause and no missing evidence
(`gate-summary.json`). The profile `mimo-v2.6-flash/cuda-spark-128g-262k` is
qualified as the switchable alias `mimo26flash`. `qwen38fn` stays the default.

## The run (one fresh start, 2026-10-13)

- **Head:** `2a75f1fe`, clean; the kit manifest's head is the same.
- **Identity:**
  - both runtime sources at their pinned revision and tree;
  - the exllamav3 wheel's 629 installed files byte-identical;
  - all 25 pack files and 3 drafter files at their manifest sha256;
  - the agent-gate fixture equal to the frozen one.
- **Production:** `qwen38fn` stopped at 22:02Z and restored at 23:17:50Z (rc 0,
  first attempt). The driver exited rc 0.
- **Start gate:** MemFree 113.87 GiB and MemAvailable 115.26 GiB.
- **Load:** launched 22:03:03Z, ready after 58 s. MemFree's low during the
  load was 9.42 GiB.

## Clauses

| # | clause | measured | result |
|---|---|---|---|
| 1 | no kill lines | kernel log (driver start to 5 s after the engine stop) empty; server and memwatch clean | pass |
| 2 | swap growth ≤ 0.25 GiB | 0.00 GiB over 3,815 samples, at most 2 s apart | pass |
| 3 | low ≥ 8 GiB | **9.56 GiB**, at 22:24:37Z (the start of the 250K fill) | pass |
| 4 | peak ≤ 106 GiB, `ceil(peak + 8)` ≤ 115 | **105.71 GiB** (115.26 − 9.56); admission 114 | pass |
| 5 | direct 262K | `PASS`: 250,135 server prompt tokens at cap 262,144, needle correct, TTFT 683.9 s | pass |
| 6 | text identity | all three greedy hashes equal preflight r2's | pass |
| 7 | tool-call ≥ 19/20 | **20/20** | pass |
| 8 | agent gate 4/4 | 4/4: repeat reused 22,272/22,511; turn 2 re-ran 100 tokens; slot-thrash 355; novel prefill 780 tok/s | pass |
| 9 | soak at the 5 GiB default floor | pass: 195 requests, 0 errors, degradation 0.3%, min 10.00 GiB | pass |
| 10 | vision vs the default's 64/100 | **68/100**, 0 errors, 4 invalid; z = +0.60 (needed ≥ −1.645) | pass |
| 11 | speed suite valid | valid at both levels | pass |

**Two disclosures:**
- **Clause 10 uses a different budget.** MiMo answered at a 4,096-token
  budget, and the default's 64 was measured at 256 (PREREG). One MiMo answer
  still reached 4,096.
- **The soak's own floor is stricter.** The soak script reports
  `qualification_eligible_floor: false` against its own 12 GiB floor. The
  PREREG holds the soak to the repository's 5 GiB default, as Flash-Next's
  c9/c10 did. The soak's minimum was 10.00 GiB.

## Reported, not gating (beside Flash-Next's c10)

These are MiMo's publishable speed numbers. They come from this kit bundle,
the qualified production path with diagnostics off.

| metric | MiMo (this run) | Flash-Next c10 |
|---|---|---|
| decode at context 0 | 21.84 tok/s | 34.07 tok/s |
| decode at 28,672 | 37.37 tok/s | 40.92 tok/s |
| prefill at 28,672 | 780 tok/s | 2,711 tok/s |
| TTFT at context 0 | 0.651 s | 0.21 s |
| TTFT at a 250K prompt | 683.9 s | — |
| soak decode median (256-token answers) | 29.72 tok/s | — |
| MMMU-val-100 | 68 | 64 |
| tool-call | 20/20 | 20/20 |
| context cap | 262,144 | 1,048,576 |

Every speed figure misses Flash-Next's. Speed targets are general, so the
misses are reported, not blocking (owner, 2026-09-11).

**Fidelity** (M1, against the release weights): ΔNLL 0.0503, KLD 0.0836,
top-1 −0.94 pp. This is a lossy 2.52 bpw pack. The numbers are reported, not
waived; adoption is the owner's decision.

**Accuracy** comes from the separately preregistered holdout draw on this
configuration.

## Review and candidates

| | candidate 1 (`d1105a04`) | candidate 2 (`d7cca682`) | candidate 3 (`2a75f1fe`) |
|---|---|---|---|
| started | no | no | yes: this run |
| open critical/high after review | 0 | 1 (C2-1, high) | 0 |

- **Round 1:** reviewer A had 1 medium and 9 lows; reviewer B had 3 mediums
  and 6 lows. All were closed in candidate 2, except B4–B6, which were
  attested.
- **Round 2:** C2-1, a regression from candidate 2's cleanup hardening, plus
  three lows. All were closed in candidate 3.
- **Round 3:** 0 open findings, apart from one optional low, which is attested
  here:
  - **A3-L1:** no mutation isolates the gate line's part of the launch
    baseline.
  - That term can only raise the baseline. In this run the gate line and the
    samples gave the same 115.26 GiB, so it did not change the verdict.
- **Convergence:** one candidate went without a decrease (0 → 1). There were
  three candidates, and each review round used the same two persistent
  reviewers.

The mutation suite passes 59/59 on the freeze.

## Evidence

- `observe/`:
  - `driver.log`;
  - 1 Hz `meminfo.log` with the scope's `anon` and `file` (anon peak 4.28 GB);
  - `kernel.log`, which is scored;
  - `kernel-restore.log`, which is recorded only: it holds one
    `drop_caches` line from the switch;
  - `server.log`, `memwatch.log`, `cgroup.log`, `restore.log`.
- **Kit bundle:** `manifest.json`, `summary.json`, `SUMMARY.md`, and the
  `speed/`, `toolcall/`, `vision/`, `context/` and `soak/` cells.
- **Gate inputs:** `probe-c1.json`, `agent-gate.json`.
- **Verdict:** `gate-summary.json`, from the frozen scorer, rerun after the
  driver exited.
