# Candidate 8 result: FAIL on clause 8 alone (soak floor harness mismatch)

The frozen scorer (`gate_verdict.py`, freeze `7a3812c0`, mutations 42/42)
returns **FAIL** in `gate-summary.json`. The only failing clause is 8, the
soak, and nothing is missing.

**The soak passed every one of its own checks except the memory floor:**
- 211 requests, 0 errors;
- decode median 30.88 tok/s, with no degradation (−7.3%);
- health always OK;
- minimum MemAvailable 9.92 GiB.

`35_soak.py` scored that minimum against its default 12 GiB floor
(`mem_floor_source: default`), not the profile's 8 GiB floor. The kit passes
`safety.kill_floor_gib` to the context cell (`--mem-floor-gib`) but nothing to
the soak. `35_soak.py` reads only `DSV4_SOAK_MEM_FLOOR_GIB`, which the owner
set to 8 for DeepSeek's 1M-fast profile on 2026-08-09. So under the kit, a
profile with an 8 GiB floor fails the soak by construction.

This PREREG did not account for that. It is an execution defect (a harness
mismatch), and candidate 9 closes it. The verdict here stays `FAIL`. Nothing
is re-scored.

## What the run measured

| Clause | Result |
|---|---|
| 1. Kill lines (kernel, server, memwatch) | none: 0 NVRM, 0 Xid, 0 OOM, no BREACH |
| 2. Swap growth | 0.0 GiB |
| 3. Low point ≥ 8 GiB | 9.62 GiB (18:21:35Z, during the 1M fill) |
| 4. Direct 1M | KV capacity 1,079,117 at 262,144; fill PASS, 1,000,672 tokens, 4/4 needles, controls OK; cap 1,048,576 |
| 5. Text path vs candidate 5 | 3/3 greedy sha256 identical |
| 6. Tool-call | 20/20 |
| 7. Agent gate | 4/4 |
| 8. Soak | **FAIL**: only `memory_above_floor` (9.92 against a default 12) |
| 9. Vision | 63/100, 3 invalid, 0 errors; z = −0.147 against 64 |
| 10. Admission | peak 105.79 GiB (launch baseline 115.41 − low 9.62); +8 = 113.79 ≤ 115 |
| 11. Precondition | 16.55 + 0.86 GiB of qwen38-1m's files resident at launch |
| 12. Fixed worker | one release under `/home/bmarti44/models`, 496 files; **MemFree 23.28 GiB** at the 17.90 GiB arena + 2 GiB |

**The fix worked.**
- **Priming reproduced c7's state.** It gave MemFree 90.8 GiB with
  23.2 GiB cached.
- **The load-time release cleared it.** Twenty seconds after launch, Cached
  had fallen to 5.8 GiB and MemFree was 104.6. In candidate 7, Cached stayed
  near 21 GiB for the whole load.
- **The arena had room.** It met 23.28 GiB of MemFree, against 10.2 in
  candidate 7, and the kernel logged no NVRM line.

**Reported, not gating** (kit `SUMMARY.md`):
- decode 25.66 tok/s at 0 and 34.70 at 28,672;
- prefill 2,678.85 tok/s at 28,672;
- TTFT 12.95 s at 28,672 and 0.30 s at 0;
- idle MemAvailable 11.25 GiB at ready.

The run ended cleanly. The driver's own stop left the scope empty, and the
`postrun` reading was 115.5 GiB with swap unchanged. No operator action was
needed.

## Next: candidate 9

The change is only what closes the soak mismatch:
- **Opt-in kit option.** A profile may declare
  `qualification_options.soak_mem_floor_gib`. The kit then runs the soak with
  `DSV4_SOAK_MEM_FLOOR_GIB` set to that value, and refuses any value below the
  profile's own `kill_floor_gib`. Profiles that declare nothing keep the
  12 GiB default.
- **This profile declares 8,** the owner's floor for it.
- **The new scorer** also requires the soak artifact to record that floor.
- **A fresh run is required.** The soak's memory depends on the vision cell
  and the fill that run before it, so a soak-only run would be weaker.
