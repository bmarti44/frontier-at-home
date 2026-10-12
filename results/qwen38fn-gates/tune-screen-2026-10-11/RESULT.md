# Flash-Next lossless lever screen: COMPLETE, keeps L1 and L2 (DIAGNOSTIC)

**Verdict** from `screen_verdict.py` (self-test 33/33; `summary.json`): `COMPLETE`.
- Kept levers: **L1** (PLE `MADV_WILLNEED`) and **L2** (65,536-row draft vocabulary plus NVFP4 draft head).
- Carrier arm: **a3**, profile `qwen3.8-flash-next/cuda-spark-128g-1m-tune-a3`.

These are single fresh-start diagnostic numbers at 3 reps. They choose candidate c10's levers. They are not headline, README or acceptance numbers: c10's matched ABBA blocks and kit bundle decide those.

## Arms (2026-10-11T23:37Z to 2026-10-12T01:56Z, production stopped throughout)

| Arm | Lever | Decision | Decode at 0 | Decode at 28,672 | Prefill at 28,672 | TTFT at 0 | Peak use | MTP accept len |
|---|---|---|---|---|---|---|---|---|
| a0 | baseline, runtime-001 | BASELINE | 25.26 | 35.50 | 2,675 | 0.274 s | 104.95 GiB | 2.73 |
| a1 | L0, runtime-002 | VOID (`ATTEMPTS.md`) | — | — | — | — | — | — |
| a1r | L0, runtime-002 | PASS: 0.993× / 1.025×, prefill 1.003× | 25.08 | 36.38 | 2,682 | 0.274 s | 105.01 GiB | 2.73 |
| a2 | L1, PLE `MADV_WILLNEED` | **KEPT:** 1.147× / 1.044×, prefill 1.002× | 28.76 | 37.98 | 2,689 | 0.212 s | 104.32 GiB | 2.75 |
| a3 | L2, draft vocab 65,536 + NVFP4 head | **KEPT:** 1.189× / 1.082×, prefill 1.008× | **34.21** | **41.10** | 2,710 | 0.212 s | 104.40 GiB | 2.57 |
| a4 | L3, k=4 at one request | DROPPED: readiness | — | — | — | — | 103.74 GiB | — |
| a5 | L4 on a4 | SUPERSEDED | — | — | — | — | — | — |
| a5b | L4, BF16 MTP KV, on a3 | DROPPED: refused at admission | — | — | — | — | — | — |

**Cumulative, a3 over a0:**
- decode **1.354×** at 0 and **1.158×** at 28,672;
- prefill 1.013×;
- TTFT 0.274 → 0.212 s;
- peak 0.56 GiB lower.

**Every measured arm was clean:**
- the three greedy probes are byte-identical to a0's and candidate 5's;
- tool-call 20/20;
- no kernel, server or memwatch kill line;
- swap growth 0;
- KV capacity 1,079,117 tokens at 262,144 per request.

The projected 1M peak of the carrier (arm peak − a0 peak + candidate 9's 105.67 GiB) is 105.12 GiB, within the 107 GiB bound.

## What each lever did

- **L0 (runtime-002: #59528, #58449, #58737) is speed-neutral.**
  - Decode moved 0.993× at 0 and 1.025× at 28,672, within single-start noise.
  - #58449 took effect: `Fused multi-step draft decode is not supported` is gone from the server log in every runtime-002 arm.
- **L1 gives most of its gain at short context.** Decode rose +14.7% at 0 and +4.4% at 28,672, and TTFT at 0 fell 23%. This matches the PLE-stall mechanism: the per-step PLE row fetch is a larger share of a short step.
- **L2 trades a lower MTP acceptance length for cheaper drafts.** Acceptance length fell 2.75 → 2.57 because the drafter can no longer propose ids ≥ 65,536, but the cheaper draft head nets +18.9% at 0 and +8.2% at 28,672. The target still verifies against its full BF16 head, and the probes are unchanged.
- **L3 does not fit 1M at the fixed arena.**
  - The k=4 schedule moved the attention block from 3,200 to 3,216 tokens and left 1,033,876 KV tokens at 262,144 per request, which is below 1,048,576.
  - The driver logged `FAILED_ATTEMPT` at readiness; there was no kill line and no swap.
  - L3 would need a larger arena, which is memory that L4 also wants. It was not pursued here.
- **L4 was not measured.** Amended arm a5b (a3 + L4) was refused by `93_profile_serve.sh`'s membudget admission before any load.
  - Its modelled budget needs 77 + 10.7 + 19.2 + 8 = 114.9 GiB. The host had 114.66 GiB at that moment.
  - The focused review had flagged the mismatch with the driver's 114.5 GiB guard as LOW.
  - A retry would not change the outcome on this host. Beyond the screen, L4 at 1M would also need c10's admission `ceil(peak + 8)` to reach 115, against a quiet host of 114.7–115.6 GiB. The switch could then refuse it on an ordinary day.
  - So L4 is not kept. It would first need a smaller-footprint form, or more host headroom.

## Execution record

- **a1 attempt 1 was voided under the preregistered rule.** It was a harness defect: stale runtime-001 page cache tripped the engine's arena guard at MemFree 19.85 GiB against 19.90 needed. `ATTEMPTS.md` has the evidence, and the fix was committed before the retry.
  - From then on, each arm's preflight releases both runtime trees' page cache.
  - a1r's arena check then passed at MemFree 21.86 GiB.
- **Two amendments, each committed before its arm loaded:**
  - a2 rebased onto a1r;
  - a5b (L4 on a3), after L3 was dropped.

## Host observations (reported)

- **MemAvailable at arm start drifted down about 0.9 GiB** during a1r: 115.43 before, about 114.75 after. It stayed flat through a2–a5b.
  - It is not an engine survivor: no process was left, and every scope was empty after its stop.
  - It is not swap, which did not grow.
  - About 1.6 GiB of memory is unaccounted by `/proc/meminfo`'s categories. That is consistent with driver-held memory, though not proven.
  - Production's own admission (`minimum_start_gib` 114) still fits.
- **The reverse switch direction applies to runtimes too.** Any start that follows a different runtime meets about 2.5 GiB of that runtime's cached libraries outside the model root. c10 has to cover this for its own run 1 and for the promotion switch:
  - the worker release is to include the runtime root;
  - `97_matched_speed_blocks.py --release-tree` covers the ABBA campaign.

## Next

**Candidate c10** is profile `tune-a3` (runtime-002 + L1 + L2), qualified under its own preregistration (plan A2):
- the c9 clause set;
- 5 matched fresh-server ABBA/BAAB blocks against the main profile with `97_matched_speed_blocks.py`;
- the runtime-root page-cache release;
- a holdout spend only if the probe hashes differ. They did not differ in this screen.
