# Candidate 9 result: PASS

The frozen scorer (`gate_verdict.py`, freeze `c73c16b0`) returns **PASS** in
`gate-summary.json`, with no failed clause and no missing evidence. Before
scoring, `gate_mutations.py` matched 43/43. The run head was `c73c16b0`,
clean.

My first scorer call omitted the bundle-directory argument. It exited with a
usage error before reading any evidence and wrote nothing. The second call,
`gate_verdict.py <this directory>`, wrote the verdict.

| Clause | Result |
|---|---|
| 1. Kill lines (kernel, server, memwatch) | none: 0 NVRM, 0 Xid, 0 OOM, no BREACH |
| 2. Swap growth | 0.0 GiB |
| 3. Low point ≥ 8 GiB | 9.63 GiB (19:51:03Z, during the 1M fill) |
| 4. Direct 1M | KV capacity 1,079,117 at 262,144; fill PASS, 1,000,672 tokens; cap 1,048,576 |
| 5. Text path vs candidate 5 | 3/3 greedy sha256 identical |
| 6. Tool-call | 20/20 |
| 7. Agent gate | 4/4 |
| 8. Soak at the 5 GiB repository floor | PASS: 211 requests, 0 errors, 1,804.6 s, decode median 30.87 tok/s, degradation −5.5%, minimum 9.91 GiB; `mem_floor_gib` 5.0 from `default` |
| 9. Vision | 63/100, 3 invalid, 0 errors; z = −0.147 against 64 |
| 10. Admission | peak 105.67 GiB (launch 115.30 − low 9.63); +8 = 113.67 ≤ 115 |
| 11. Precondition | 16.55 + 0.86 GiB of qwen38-1m's files resident at launch |
| 12. Fixed worker | one release under `/home/bmarti44/models`, 496 files; MemFree 22.75 GiB at the 17.90 GiB arena + 2 GiB |

**The page-cache fix held under the adverse precondition, for the second
time.**
- Priming left MemFree at 87.8 GiB with 26.0 GiB cached.
- The worker's release brought the arena's MemFree to 22.75 GiB. Candidate 7
  had 10.2 GiB there.
- The kernel log is empty for the whole start.

## Reported, not gating

From the kit's `SUMMARY.md` (kit verdict `PASS`):
- **Decode:** 25.76 tok/s at 0 and 34.89 at 28,672.
- **Prefill:** 2,683.74 tok/s at 28,672.
- **TTFT:** 0.27 s at 0 and 12.93 s at 28,672.
  - At 0 this meets the 0.5 s target. Candidate 5 measured 0.57 s, the one
    target it missed.
- **Agent gate prefill:** 2,692 tok/s on a novel prompt.

**`qualification_eligible_floor` is `false` in `soak/soak.json`. That is
expected and does not bear on this gate.**
- The flag records whether a soak is admissible to the frozen 2026-08 DSV4
  decision procedure (`scripts/34_decision.py`). That procedure fixes a 12 GiB
  floor.
- This soak ran at the repository-wide 5 GiB floor (owner, 2026-10-11), which
  is what clause 8 requires.

The driver logged `ERROR: ... is not running (state file absent)` after
`scope empty after stop`. That is its post-stop `93_profile_serve.sh --stop`,
which finds the server already gone, as in candidate 8. The scope was empty,
and the `postrun` reading was 115.5 GiB with swap unchanged.

## Review lows carried into the run (round on `c73c16b0`: no critical or high)

- **An inherited `DSV4_SOAK_MEM_FLOOR_GIB` would have wasted the run as
  `NO_RESULT`.** The driver ran under `env -u DSV4_SOAK_MEM_FLOOR_GIB`, and
  the soak recorded source `default`.
- **The comment at `scripts/34_decision.py:109` still says its constants equal
  `35_soak.py`'s.** The 12 GiB constant no longer equals the soak default.
  - The file is MANIFEST-frozen, and its behaviour fails closed: it rejects a
    5 GiB soak.
  - `35_soak.py` documents the split.
  - Left for a later non-gate commit.
- **No mutation case covers 5.0 with `env-override`.** The scorer rejects it
  through its source check. Coverage only; the count stays as preregistered.

## Admission derived from this run (for the promotion commit)

- `memory_model.overhead_gib` = 105.67 − 77 − 18.0 = 10.67, rounded up to
  **10.7**.
- `safety.minimum_start_gib` = ceil(105.67 + 8) = **114**.
