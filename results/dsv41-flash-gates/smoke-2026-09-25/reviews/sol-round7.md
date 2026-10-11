## Findings

1. **HIGH — Finding 3 is PARTIALLY CLOSED.** Verbatim replay now fails, and the bundler rehashes both external fixtures. But [phase_timing()](</home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/frozen_check.py:110>) trusts editable `phase`, `recorded_at`, and file mtime. I copied one genuine launch record into all three phases, changed only those values, and `verify()` returned `(True, [])`. A replay can therefore stand in for checks that never ran after execution or scoring. **Minimal fix:** require an independently timestamped, tamper-evident receipt for each phase and verify it at bundling.

2. **HIGH — New false-failure risk from fixed time windows.** [frozen_check.py:120](</home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/frozen_check.py:120>) allows 60 seconds from launch check to start and 600 seconds from finish to post-run check or from post-run check to post-score check. The [runner](</home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/run_smoke.sh:91>) performs identity verification before start, then crashlog copying, `journalctl`, and another identity check before post-run verification; fidelity scoring follows it. None has a matching time bound. A slow but genuine step can make an unchanged run FAIL. **Minimal fix:** check phase order against recorded step boundaries without imposing unverified maximum durations; test delayed legitimate steps.

The stdout redirection does **not** inherently cause an mtime failure: the JSON write updates the file mtime after `recorded_at` ([run_smoke.sh:68](</home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/run_smoke.sh:68>)). The [fixture-drift test](</home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/test_bundle_attempt.py:303>) is sound for the rehash check: it substitutes a changed fixture copy and confirms rejection.

| Item | Status |
| --- | --- |
| Finding 3 | **PARTIALLY CLOSED** |
| New critical/high defects | **One HIGH:** timing windows can reject genuine runs |
| Preregistered text arm, then 112-case fidelity smoke | **Not fit for authoritative execution** with these findings open. This is a fidelity smoke, not a capability result. |