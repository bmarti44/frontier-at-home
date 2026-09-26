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
