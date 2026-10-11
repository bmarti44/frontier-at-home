# Smoke gate paused (2026-09-26): convergence rule

The gate has run 7 candidates, and this was campaign review round 6 (GPT-6 sol, high reasoning, read-only). The reviews are in `reviews/sol-round{1..6}.md`.

## Open HIGH and CRITICAL findings per round

| Round | Candidate reviewed | Open HIGH/CRITICAL |
| --- | --- | ---: |
| 1 | 2 | 6 |
| 2 | 3 | 3 |
| 3 | 4 | 3 |
| 4 | 5 | 1 |
| 5 | 6 | 1 |
| 6 | 7 | 1 |

Blocking findings did not strictly decrease across two consecutive candidates (6 and 7). Under AGENTS.md that pauses the gate for an owner report. No acceptance arm has run, and no model was loaded during the review loop.

## Open finding (HIGH, finding 3 remainder, round 6)

`frozen_check.verify()` does not bind each record's `phase` to the file it is stored in. It also does not rehash the external fixture files (the fixture manifest and the Metal reference) when bundling. So a launch-time record copied into `frozen-check-post-run.json` and `frozen-check-post-score.json` passes: sol reproduced `(True, [])` for `[r, r, r]`. A stale launch check can therefore stand in for the checks after the run and after scoring.

Sol's minimal fix:
- require the exact phase for each filename;
- rehash every frozen path, including the external fixtures, at bundling time;
- add replay and fixture-drift mutation tests.

Inherited limitation, disclosed: a commit that moves HEAD during a run makes otherwise-valid phase records disagree on the commit, so the run FAILs. That errs toward failing.

No new critical or high defect was found in candidate 7.

## Update: candidate 8 and review round 7 (2026-09-26)

The owner authorized "the candidate 8 fix, then one more review". Candidate 8 is commits `a51c8fe2` and `636be6fd`, plus the preregistration in `6933f704`. Review round 7 is `reviews/sol-round7.md`, and its verdict is **not fit**. The gate is paused again. It has run 8 candidates, which is the per-gate maximum, across 7 campaign review rounds.

| Round | Candidate | Open HIGH/CRITICAL |
| --- | --- | ---: |
| 7 | 8 | 2 |

The open findings:

1. **HIGH, finding 3 still partially closed.** Verbatim replay now fails, and both fixtures are re-hashed at bundling. But a genuine launch record copied into all three phases passes `verify()` if its editable `phase`, `recorded_at` and file mtime are changed. Sol's fix is an independently timestamped, tamper-evident receipt for each phase, checked at bundling.
2. **HIGH, new in candidate 8.** The fixed 60 s and 600 s windows could reject a genuine run whose identity check, `journalctl` step or scoring runs slowly. This errs toward a false FAIL, not a false PASS. Sol's fix is to check the order of the phases against recorded step boundaries, without unverified maximum durations.

No acceptance arm has run. The next step is the owner's decision.

## Resolution: owner decision (2026-09-26)

The owner judged the review loop over-engineered and chose the following:

- **Candidate 9 is discarded.** It was never committed. It would have added journald receipts.
- **Forgery is out of scope.** Round 7 finding 1 was forgery of evidence by someone with write access, and it is recorded as out of scope. AGENTS.md now says so explicitly under "Keep process proportionate".
- **Round 7 finding 2 is an accepted known limitation.** A slow step can exceed a fixed time window, which can produce only a visible false FAIL, never a false PASS.
- **Candidate 8 runs as a labeled fidelity smoke.** It is diagnostic evidence, not an acceptance result.
- **Later gates use the shared tooling** (`scripts/94_qualify_profile.py` and the profile system), not more per-model harness code.

The gate is closed. No more review rounds will be spent on this harness.
