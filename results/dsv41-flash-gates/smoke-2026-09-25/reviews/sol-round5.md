## Finding

**HIGH — The bundler accepts forged frozen-check records.** [bundle_attempt.py:130](/home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/bundle_attempt.py:130) requires the phases to agree, but never verifies that their commit exists, that it contains the frozen-inputs file, or that the hash map includes all ten components. Three records containing `ok: true`, the nonexistent commit `111…111`, and only the current scorer and frozen-file hashes pass `frozen_consistent`. I confirmed that result without changing files. The test fixture itself uses that abbreviated map and invented commit ([test_bundle_attempt.py:42](/home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/test_bundle_attempt.py:42)). **Minimal fix:** verify the recorded commit against Git, compare its frozen-inputs blob with the file, and require the complete expected component map and per-path check results; add a mutation test for this forgery.

The chained comparisons at [bundle_attempt.py:141](/home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/bundle_attempt.py:141) have the intended Python meaning: each requires both hash equality and a non-`None` file hash. They are not a separate defect.

| Review item | Status |
| --- | --- |
| Round-4 finding 3 | **PARTIALLY CLOSED.** The runner now checks the committed blob at each phase ([run_smoke.sh:91](/home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/run_smoke.sh:91)); the bundler still accepts the forgery above. |
| Round-4 findings 1, 2, 4 | **CLOSED**, as previously reviewed; candidate 6 does not interact with their closure. |
| New critical defects | None found. |
| Preregistered text arm, then 112-case fidelity smoke | **Not fit for authoritative execution** until finding 3 is closed. This is a fidelity smoke, not a capability result. |

The named-path Git status check excludes both the untracked attempt directory and the unrelated `mla-replay-001/` directory ([run_smoke.sh:90](/home/bmarti44/spark-deepseek-v4-flash/results/dsv41-flash-gates/harness/run_smoke.sh:90)). The committed blob and current frozen hashes match. With unchanged files and `HEAD`, a genuine fidelity run would supply consistent hashes across all three phases and pass this new consistency check.