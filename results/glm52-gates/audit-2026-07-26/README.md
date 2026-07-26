# Adversarial re-audit (2026-07-26)

Brian directive: one sol xhigh agent per finding, each tasked to **prove the
claim false** rather than merely review it. Every agent writes its verdict to
`<finding-slug>.md` in this directory. `INDEX.md` carries the synthesis
(SURVIVED / WEAKENED / FALSIFIED per finding) and any ledger corrections.

Evidence under audit lives in `results/glm52-gates/loadprof-2026-07-25.json`,
`G4-bench.json`, `logs/loadprof1/`, and the harnesses in `harness/`.
